import PhotosUI
import SwiftUI

/// The app: the start (the saved work and the keys that bring in more) or the editor, with full
/// pages over them. The editor: the top bar, the picture on the app's own background, the play row,
/// the timeline (or for photos, the photo strip and its faces), and the row of tools. A tool's
/// settings, and the export, take the timeline's place; with segments selected the tools become the
/// clip tools.
struct EditorView: View {
    @State private var model = EditorModel()
    @State private var pages: [Page] = []
    /// which way the tools last moved, so a panel comes in from that side
    @State private var toolDirection: CGFloat = 0
    @State private var undoTaps = 0
    @State private var redoTaps = 0
    @State private var videoItem: PhotosPickerItem?
    @State private var photoItems: [PhotosPickerItem] = []
    @State private var characterItem: PhotosPickerItem?
    /// the characters panel is removing and renaming
    @State private var editingCharacters = false
    @State private var renaming: Int?
    @State private var draft = ""
    @FocusState private var typing: Bool
    @Environment(\.scenePhase) private var phase

    var body: some View {
        GeometryReader { geo in
            ZStack {
                editor
                    .offset(x: pages.isEmpty ? 0 : -geo.size.width * 0.26)
                    .opacity(pages.isEmpty ? 1 : 0.45)
                    .allowsHitTesting(pages.isEmpty)
                ForEach(Array(pages.enumerated()), id: \.element) { index, page in
                    pageView(page)
                        .offset(x: index < pages.count - 1 ? -geo.size.width * 0.26 : 0)
                        .transition(.move(edge: .trailing))
                        .zIndex(Double(index) + 1)
                }
            }
        }
        .background(Ink.bg.ignoresSafeArea())
        .photosPicker(isPresented: picker(.video), selection: $videoItem, matching: .videos, photoLibrary: .shared())
        .photosPicker(isPresented: picker(.photos), selection: $photoItems, maxSelectionCount: 50, matching: .images, photoLibrary: .shared())
        .photosPicker(isPresented: picker(.morePhotos), selection: $photoItems, maxSelectionCount: 50, matching: .images, photoLibrary: .shared())
        .photosPicker(isPresented: picker(.character), selection: $characterItem, matching: .images, photoLibrary: .shared())
        .onChange(of: videoItem) { _, item in
            guard let item else { return }
            videoItem = nil
            Task { await model.importVideo(item) }
        }
        .onChange(of: photoItems) { _, items in
            guard !items.isEmpty else { return }
            let adding = model.work != nil
            photoItems = []
            Task { await model.importPhotos(items, adding: adding) }
        }
        .onChange(of: characterItem) { _, item in
            guard let item else { return }
            characterItem = nil
            Task {
                if let data = try? await item.loadTransferable(type: Data.self), let image = UIImage(data: data) {
                    await model.addCharacter(image)
                } else {
                    model.show("That picture can't be opened. Try another.")
                }
            }
        }
        .onChange(of: phase) { _, now in
            model.live?.setActive(now == .active)
            ModelFiles.shared.setActive(now == .active)
            if now != .active { model.save() }
        }
        .onChange(of: model.work?.id) { _, _ in
            editingCharacters = false
            renaming = nil
        }
    }

    private func picker(_ kind: EditorModel.Picking) -> Binding<Bool> {
        Binding(get: { model.picking == kind }, set: { if !$0 && model.picking == kind { model.picking = nil } })
    }

    private func push(_ page: Page) {
        withAnimation(Motion.ease(Motion.move)) { pages.append(page) }
    }

    private func pop() {
        withAnimation(Motion.ease(Motion.move)) { _ = pages.popLast() }
    }

    @ViewBuilder
    private func pageView(_ page: Page) -> some View {
        switch page {
        case .settings: SettingsPage(model: model, onBack: pop, open: push)
        case .models:
            ModelsPage {
                pop()
                // (what a model that's come does shows on the picture)
                model.live?.refreshMade()
                model.live?.refreshPicture()
            }
        case .licenses: TextPage(title: "Licenses", paragraphs: Self.licenses, onBack: pop)
        case .responsibly: TextPage(title: "Use it responsibly", paragraphs: Self.responsibly, onBack: pop)
        }
    }

    private static let licenses = [
        "Models: InsightFace inswapper_128, SCRFD-10G and ArcFace R50 (non-commercial research license); GPEN face enhancer (Yang et al., no stated license); community-trained XSeg occlusion mask, on the architecture of DeepFaceLab (GPL-3.0).",
        "FaceJutsu doesn't include these models; you download them yourself. Some are for non-commercial research only and one states no license, so check each license before using them or what you make with them.",
        "FaceJutsu is free software under the AGPL-3.0.",
    ]

    private static let responsibly = [
        "Only swap the faces of people who have agreed to it. Don't use FaceJutsu to deceive, harass or impersonate anyone, or to make sexual images of anyone without their consent, or of minors at all. What you make and share is your responsibility.",
    ]

    private var editor: some View {
        VStack(spacing: 0) {
            topBar
            if model.work == nil {
                StartView(model: model)
                    .overlay(alignment: .bottom) {
                        if let toast = model.toast {
                            Toast(text: toast)
                                .padding(.bottom, 150)
                                .transition(NoCatchFade())
                                .onTapGesture { withAnimation(Motion.ease(Motion.quick)) { model.toast = nil } }
                        }
                    }
            } else {
                StageView(model: model)
                    .frame(maxHeight: .infinity)
                    .onTapGesture { closeTool() }
                    .allowsHitTesting(!model.exportRunning)
                transport
                ZStack(alignment: .bottom) {
                    if model.mode == .video {
                        TimelineView(model: model)
                    } else {
                        PhotoPane(model: model)
                    }
                    if let export = model.export {
                        ZStack { Ink.bg; exportPanel(export) }
                            .transition(NoCatchFade())
                    } else if let tool = model.tool {
                        // one surface for all the tools: only what's on it changes, so the timeline
                        // never shows between two of them
                        ZStack {
                            Ink.bg
                            panel(tool).id(tool).transition(.identity)
                        }
                        .transition(NoCatchFade())
                    }
                    if let toast = model.toast {
                        Toast(text: toast)
                            .transition(NoCatchFade())
                            .onTapGesture { withAnimation(Motion.ease(Motion.quick)) { model.toast = nil } }
                    }
                }
                .frame(height: Metric.timeline)
                .animation(Motion.ease(Motion.quick), value: model.export != nil)
                tools
                    .opacity(model.exportRunning ? 0.35 : 1)
                    .allowsHitTesting(!model.exportRunning)
            }
        }
    }

    private func closeTool() {
        guard model.tool != nil else { return }
        withAnimation(Motion.ease(Motion.quick)) { model.tool = nil }
    }

    // MARK: the two bars

    private var topBar: some View {
        HStack(spacing: 0) {
            if model.work != nil {
                Button { withAnimation(Motion.ease(Motion.move)) { model.close() } } label: { IconView(.close) }
                    .buttonStyle(IconButtonStyle(color: Ink.fg))
                    .padding(.leading, -Metric.gutter)
                    .padding(.trailing, -2)
                    .opacity(model.exportRunning ? 0.35 : 1)
                    .disabled(model.exportRunning)
                    .accessibilityLabel("Close")
            }
            HStack(spacing: 6) {
                Mark(size: 24, paper: Ink.bg, wink: model.poofTick).foregroundStyle(Ink.fg)
                Wordmark(size: 17).foregroundStyle(Ink.fg)
            }
            Spacer(minLength: 4)
            if model.work != nil {
                Button { model.startExport() } label: {
                    HStack(spacing: 6) { IconView(.export, size: 16, weight: 2); Text("Export") }
                }
                .buttonStyle(PrimaryButtonStyle())
                .opacity(model.canExport && !model.exportRunning ? 1 : 0.35)
                .disabled(!model.canExport || model.exportRunning)
            }
            Button { push(.settings) } label: {
                IconView(.gear).rotationEffect(.degrees(pages.isEmpty ? 0 : 70))
            }
            .buttonStyle(IconButtonStyle())
            .opacity(model.exportRunning ? 0.35 : 1)
            .disabled(model.exportRunning)
            .accessibilityLabel("Settings")
        }
        .padding(.leading, Metric.gutter)
        .padding(.trailing, 4)
        .frame(height: Metric.hit)
    }

    // The same grid as the top bar: the time starts where the wordmark does, the last icon sits under
    // the gear, and play is exactly on the playhead.
    private var transport: some View {
        let running = model.exportRunning
        return ZStack {
            HStack(spacing: 0) {
                if model.mode == .video {
                    Text(EditorModel.timecode(model.playhead, hundredths: true))
                        .font(.mono(12, 500)).foregroundStyle(Ink.fg).monospacedDigit()
                    Text(" / \(EditorModel.timecode(model.duration))")
                        .font(.mono(12, 500)).foregroundStyle(Ink.faint)
                }
                Spacer(minLength: 0)
                if !running {
                    if model.mode == .photo && !model.doc.photos.isEmpty {
                        Button { model.removePhoto() } label: { IconView(.trash, size: 19, weight: 1.75) }
                            .buttonStyle(IconButtonStyle(color: Ink.fg, height: 36))
                            .accessibilityLabel("Remove this photo")
                    }
                    Button { model.deselect() } label: { IconView(.deselect) }
                        .buttonStyle(IconButtonStyle(color: Ink.fg, height: 36))
                        .opacity(model.hasSelection ? 1 : 0)
                        .scaleEffect(model.hasSelection ? 1 : 0.6)
                        .animation(Motion.ease(Motion.quick), value: model.hasSelection)
                        .disabled(!model.hasSelection)
                        .accessibilityLabel("Clear the selection")
                    Button { undoTaps += 1; model.undo() } label: {
                        IconView(.undo).iconAct(.swing(-32), on: undoTaps)
                    }
                    .buttonStyle(IconButtonStyle(color: Ink.fg, height: 36))
                    .disabled(!model.canUndo)
                    .opacity(model.canUndo ? 1 : 0.4)
                    .accessibilityLabel("Undo")
                    Button { redoTaps += 1; model.redo() } label: {
                        IconView(.redo).iconAct(.swing(32), on: redoTaps)
                    }
                    .buttonStyle(IconButtonStyle(color: Ink.fg, height: 36))
                    .disabled(!model.canRedo)
                    .opacity(model.canRedo ? 1 : 0.4)
                    .accessibilityLabel("Redo")
                }
            }
            .padding(.leading, Metric.gutter)
            .padding(.trailing, 4)
            if !running {
                // a hard cut, on purpose: comparing needs the two pictures with nothing between them
                Button { model.toggleBefore() } label: {
                    IconView(.compare, size: 20, weight: 1.75)
                        .foregroundStyle(model.before ? Ink.keyFg : Ink.fg)
                        .frame(width: 32, height: 26)
                        .background {
                            if model.before {
                                cornerShape.fill(Ink.keyBg).background { cornerShape.fill(Ink.keyEdge).offset(y: Metric.edgeSmall) }
                            }
                        }
                        .frame(width: Metric.hit, height: 36)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .transaction { $0.animation = nil }
                .offset(x: -44)
                .accessibilityLabel("Before and after")
                .accessibilityValue(model.before ? "Before" : "After")
            }
            if model.mode == .video && !running {
                Button { model.togglePlay() } label: {
                    KeyframeAnimator(initialValue: CGFloat(1), trigger: model.playing) { p in
                        IconView(model.playing ? .pause : .play, size: 15)
                            .scaleEffect(0.4 + 0.6 * p)
                            .rotationEffect(.degrees(Double(1 - p) * 90))
                            .opacity(Double(p))
                    } keyframes: { _ in
                        KeyframeTrack(\.self) {
                            MoveKeyframe(CGFloat(0))
                            CubicKeyframe(CGFloat(1), duration: Motion.quick)
                        }
                    }
                }
                .buttonStyle(KeyButtonStyle())
                .offset(y: -1)
                .accessibilityLabel(model.playing ? "Pause" : "Play")
            }
        }
        .frame(height: 36)
    }

    // MARK: the tools

    private var tools: some View {
        let items: [ToolItem]
        if model.mode == .video && model.hasSelection && !model.settingsRow {
            items = [
                ToolItem(id: "back", icon: .back, move: .nudge, narrow: true),
                ToolItem(id: Tool.characters.rawValue, icon: .person, label: "Character", move: .hop),
                ToolItem(id: "split", icon: .split, label: "Split", enabled: model.canSplit, move: .snip),
                ToolItem(id: "join", icon: .join, label: "Join", enabled: model.canJoin, move: .squeeze),
                ToolItem(id: "merge", icon: .merge, label: "Merge", enabled: model.canMerge, move: .nudge),
                ToolItem(id: "delete", icon: .trash, label: "Delete", move: .turn(9)),
            ]
        } else {
            items = [
                ToolItem(id: Tool.characters.rawValue, icon: .person, label: model.hasSelection ? "Character" : "Characters", move: .hop),
                ToolItem(id: Tool.swap.rawValue, icon: .swap, label: "Swap", seal: "1", move: .slide),
                ToolItem(id: Tool.enhance.rawValue, icon: .enhance, label: "Enhance", seal: "2", move: .sparkle),
                ToolItem(id: Tool.mask.rawValue, icon: .mask, label: "Mask", seal: "3", move: .tilt),
            ]
        }
        return ToolRow(items: items, selected: model.tool?.rawValue) { id in
            if let tool = Tool(rawValue: id) {
                let order = Tool.allCases
                if let was = model.tool, let a = order.firstIndex(of: was), let b = order.firstIndex(of: tool) {
                    toolDirection = b > a ? 1 : -1
                } else {
                    toolDirection = 0
                }
                editingCharacters = false
                withAnimation(Motion.ease(Motion.quick)) { model.tool = model.tool == tool ? nil : tool }
                return
            }
            model.tool = nil
            switch id {
            case "back": model.settingsRow = true
            case "split": model.split()
            case "join": model.join()
            case "merge": model.merge()
            case "delete": model.delete()
            default: break
            }
        }
        .overlay(alignment: .topLeading) {
            // where to go next, on a first video with no characters
            if model.doc.characters.isEmpty && model.tool == nil && !model.hasSelection && model.export == nil {
                GeometryReader { geo in
                    markShape.fill(Ink.accent)
                        .frame(width: 9, height: 9)
                        .rotationEffect(.degrees(-4))
                        .offset(x: 4 + (geo.size.width - 8) / 8 + 10, y: 7)
                }
                .allowsHitTesting(false)
            }
        }
    }

    /// A row of a panel, coming in from the side the tools moved.
    private func row<V: View>(_ index: Int, @ViewBuilder _ content: () -> V) -> some View {
        content().rise(index, dx: toolDirection * 18, dy: toolDirection == 0 ? 6 : 0)
    }

    @ViewBuilder
    private func panel(_ tool: Tool) -> some View {
        let s = model.scopeSettings
        switch tool {
        case .characters:
            charactersPanel
        case .swap:
            ToolPanel(seal: "1", title: "Swap", summary: s.detail, onClose: closeTool) {
                row(0) {
                    OptionRow(label: "Detail") {
                        SegKeys(options: ["Fast", "Balanced", "Max"].map { ($0, $0) }, selection: model.binding(\.detail))
                    }
                }
            }
        case .enhance:
            ToolPanel(seal: "2", title: "Enhance", summary: "\(s.quality) · \(String(format: "%.2f", s.strength))", onClose: closeTool) {
                row(0) {
                    OptionRow(label: "Quality") {
                        SegKeys(options: ["Off", "Fast", "Best"].map { ($0, $0) }, selection: model.binding(\.quality))
                    }
                }
                if ModelFiles.shared.enhancerHere(s) {
                    row(1) {
                        OptionRow(label: "Style") {
                            SegKeys(options: ["Subtle", "Strong"].map { ($0, $0) }, selection: model.binding(\.style))
                        }
                    }
                    row(2) {
                        OptionRow(label: "Strength", value: String(format: "%.2f", s.strength)) {
                            SealSlider(value: model.binding(\.strength, drag: true), label: "Strength")
                        }
                    }
                } else {
                    row(1) { modelNeeded }
                }
            }
        case .mask:
            ToolPanel(seal: "3", title: "Mask", summary: "Edges \(String(format: "%.2f", s.edges))", onClose: closeTool) {
                row(0) {
                    OptionRow(label: "Edges", value: String(format: "%.2f", s.edges)) {
                        SealSlider(value: model.binding(\.edges, drag: true), label: "Edges")
                    }
                }
                row(1) {
                    OptionRow(label: "Objects") {
                        Text("Hands and glasses stay in front")
                            .font(.ui(12)).foregroundStyle(Ink.faint).lineLimit(1)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        SealSwitch(isOn: model.binding(\.objects), label: "Objects")
                            .padding(.trailing, -3)
                    }
                }
                if ModelFiles.shared.objectsHere(s) {
                    row(2) {
                        OptionRow(label: "Coverage", value: signed(s.coverage)) {
                            SealSlider(value: model.binding(\.coverage, drag: true), range: -1...1, label: "Coverage")
                        }
                    }
                } else {
                    row(2) { modelNeeded }
                }
            }
        }
    }

    /// In a tool's panel, in place of the settings that can't do anything without their model.
    private var modelNeeded: some View {
        Text("This needs a model you don't have yet. Get it in Settings.")
            .font(.ui(12))
            .foregroundStyle(Ink.muted)
            .frame(maxWidth: .infinity, minHeight: Metric.hit, alignment: .leading)
    }

    // MARK: characters

    @ViewBuilder
    private var charactersPanel: some View {
        let current = model.scopeCharacter
        if let id = renaming, let c = model.character(id) {
            renameRow(c)
        } else if editingCharacters {
            ToolPanel(title: "Edit characters", onClose: { withAnimation(Motion.ease(Motion.quick)) { editingCharacters = false } }, closeLabel: "Done") {
                characterGrid(edit: true, current: current)
            }
        } else {
            ToolPanel(title: model.hasSelection ? "Character" : "Characters",
                      summary: model.doc.characters.isEmpty ? "None yet" : current.mixed ? "Mixed" : (model.character(current.id)?.name ?? "None"),
                      onClose: closeTool,
                      extra: model.doc.characters.isEmpty ? nil : AnyView(
                          Button { withAnimation(Motion.ease(Motion.quick)) { editingCharacters = true } } label: { IconView(.pen, size: 16, weight: 2) }
                              .buttonStyle(IconButtonStyle(color: Ink.muted, height: 32))
                              .accessibilityLabel("Edit characters")
                      )) {
                if model.doc.characters.isEmpty {
                    HStack(spacing: 6) {
                        Button { model.picking = .character } label: { AddCard() }
                            .buttonStyle(ToolPressStyle())
                            .frame(width: (UIScreen.main.bounds.width - Metric.gutter * 2 - 30) / 6)
                        Text("Add a photo of the face you want.")
                            .font(.ui(14))
                            .foregroundStyle(Ink.muted)
                            .frame(maxWidth: .infinity)
                            .multilineTextAlignment(.center)
                    }
                    .padding(.top, 6)
                } else {
                    characterGrid(edit: false, current: current)
                }
            }
        }
    }

    private func characterGrid(edit: Bool, current: (id: Int?, mixed: Bool)) -> some View {
        ScrollView(.vertical, showsIndicators: false) {
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 6), count: 6), spacing: 6) {
                ForEach(Array(model.doc.characters.enumerated()), id: \.element.id) { i, c in
                    row(i) {
                        if edit {
                            CharacterCard(name: c.name, image: c.image, editing: true)
                                .overlay(alignment: .topTrailing) {
                                    Button { withAnimation(Motion.ease(Motion.quick)) { model.removeCharacter(c.id) } } label: {
                                        IconView(.close, size: 11, weight: 3)
                                            .foregroundStyle(Ink.accentInk)
                                            .frame(width: 18, height: 18)
                                            .background { markShape.fill(Ink.accent) }
                                            .rotationEffect(.degrees(-4))
                                            .padding(3)
                                            .frame(width: 36, height: 36, alignment: .topTrailing)
                                            .contentShape(Rectangle())
                                    }
                                    .buttonStyle(.plain)
                                    .accessibilityLabel("Remove \(c.name)")
                                }
                                .onTapGesture {
                                    draft = c.name
                                    renaming = c.id
                                }
                        } else {
                            Button { model.give(c.id) } label: {
                                CharacterCard(name: c.name, image: c.image, chosen: !current.mixed && current.id == c.id)
                            }
                            .buttonStyle(ToolPressStyle())
                        }
                    }
                }
                row(model.doc.characters.count) {
                    Button { model.picking = .character } label: { AddCard() }
                        .buttonStyle(ToolPressStyle())
                }
            }
            .padding(.top, 6)
            .padding(.bottom, 6)
        }
    }

    /// Typing a character's name: one row over the keyboard, with the pen and its ink line.
    private func renameRow(_ c: SwapCharacter) -> some View {
        ToolPanel(title: "Rename", onClose: commitRename, closeLabel: "Done") {
            HStack(spacing: 8) {
                Group {
                    if let image = c.image { Image(uiImage: image).resizable().scaledToFill() } else { Ink.panel2 }
                }
                .frame(width: 36, height: 48)
                .clipShape(cornerShape)
                IconView(.pen, size: 16, weight: 2).foregroundStyle(Ink.fg)
                ZStack(alignment: .bottomLeading) {
                    TextField("Name", text: $draft)
                        .font(.ui(17, 600))
                        .foregroundStyle(Ink.fg)
                        .tint(Ink.accent)
                        .submitLabel(.done)
                        .focused($typing)
                        .onSubmit(commitRename)
                        .onChange(of: draft) { _, v in if v.count > 24 { draft = String(v.prefix(24)) } }
                        .frame(height: 40)
                    RoundedRectangle(cornerRadius: 1).fill(Ink.accent).frame(height: 2).padding(.bottom, 3)
                }
            }
            .padding(.top, 6)
            .task { typing = true }
        }
    }

    private func commitRename() {
        if let id = renaming { model.renameCharacter(id, to: draft) }
        typing = false
        renaming = nil
    }

    // MARK: the export

    @ViewBuilder
    private func exportPanel(_ e: ExportState) -> some View {
        switch e {
        case .scouting(let done, let total):
            exportRunning(done: done, total: total, note: "", summary: "Step 1 of 2 · Scouting faces", hint: "Finding everyone first. Swapping starts after this.")
        case .running(let done, let total, let left):
            exportRunning(done: done, total: total, note: left.map { "About \(max(1, Int($0.rounded()))) s left" } ?? "",
                          summary: model.mode == .video ? "Step 2 of 2 · Swapping" : "Swapping", hint: "Keep FaceJutsu open while it swaps.")
        case .paused(let done, let total):
            exportRunning(done: done, total: total, note: "", summary: "Paused", hint: "Paused while FaceJutsu was away. It carries on from here.")
        case .finished(let facts, let savedTo, let file):
            ToolPanel(check: true, title: "Jutsu complete", summary: savedTo) {
                Text(facts).font(.mono(12, 500)).foregroundStyle(Ink.muted).padding(.top, 12)
                Spacer(minLength: 0)
                HStack(spacing: 10) {
                    Spacer()
                    Button("Done") { model.closeExport() }.buttonStyle(TextButtonStyle())
                    if let file {
                        ShareLink(item: file) {
                            HStack(spacing: 6) { IconView(.share, size: 16, weight: 2); Text("Share") }
                        }
                        .buttonStyle(PaperButtonStyle())
                    }
                }
                .padding(.bottom, 10)
            }
        case .failed(let done, let total):
            ToolPanel(title: "Couldn't finish", titleColor: Ink.bad, summary: total > 0 ? "\(done * 100 / total)%" : "") {
                VStack(alignment: .leading, spacing: 2) {
                    Text("The swap stopped partway through.").font(.ui(14)).foregroundStyle(Ink.fg)
                    Text("Nothing was saved.").font(.ui(12)).foregroundStyle(Ink.faint)
                }
                .padding(.top, 6)
                Spacer(minLength: 0)
                HStack(spacing: 10) {
                    Spacer()
                    Button("Close") { model.closeExport() }.buttonStyle(TextButtonStyle())
                    Button("Try again") { model.closeExport(); model.startExport() }.buttonStyle(PrimaryButtonStyle())
                }
                .padding(.bottom, 8)
            }
        case .noRoom(let need, let free):
            ToolPanel(title: "Not enough room", titleColor: Ink.bad) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Free up \(max(1, need - free)) MB, or pick a lower quality.").font(.ui(14)).foregroundStyle(Ink.fg)
                    Text("This export needs about \(need) MB.").font(.ui(12)).foregroundStyle(Ink.faint)
                }
                .padding(.top, 6)
                Spacer(minLength: 0)
                HStack(spacing: 10) {
                    Spacer()
                    Button("Close") { model.closeExport() }.buttonStyle(TextButtonStyle())
                    Button("Settings") { model.closeExport(); push(.settings) }.buttonStyle(PaperButtonStyle())
                }
                .padding(.bottom, 8)
            }
        }
    }

    private func exportRunning(done: Int, total: Int, note: String, summary: String, hint: String) -> some View {
        let p = total > 0 ? Double(done) / Double(total) : 0
        return ToolPanel(title: "Export", summary: summary) {
            HStack(alignment: .bottom, spacing: 14) {
                Text("\(Int(p * 100))%").font(.mono(38, 500)).foregroundStyle(Ink.fg).monospacedDigit()
                VStack(alignment: .leading, spacing: 2) {
                    Text(model.mode == .video ? "\(done) of \(total) frames" : "\(done) of \(total) photos")
                    if !note.isEmpty { Text(note) }
                }
                .font(.ui(12, 500))
                .foregroundStyle(Ink.muted)
                .monospacedDigit()
                .padding(.bottom, 3)
            }
            .frame(height: 52, alignment: .bottom)
            ProgressBar(value: p).padding(.top, 8)
            HStack {
                Text(hint).font(.ui(12)).foregroundStyle(Ink.faint).lineLimit(2)
                Spacer(minLength: 8)
                StopButton { model.stopExport() }
            }
        }
    }

    private func signed(_ v: Double) -> String {
        abs(v) < 0.005 ? "0.00" : String(format: "%+.2f", v)
    }
}

private struct ToolPressStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label.pressDip(configuration.isPressed)
    }
}

/// Stop, which asks first: the first tap turns it into "Stop ?", the second stops.
private struct StopButton: View {
    let action: () -> Void
    @State private var armed = false

    var body: some View {
        Button {
            if armed {
                armed = false
                action()
            } else {
                armed = true
                Task { @MainActor in
                    try? await Task.sleep(for: .milliseconds(2400))
                    armed = false
                }
            }
        } label: {
            Text(armed ? "Stop ?" : "Stop").frame(minWidth: 44, alignment: .trailing)
        }
        .buttonStyle(TextButtonStyle(danger: true))
    }
}

// MARK: - The stage

/// The picture at the playhead (or the chosen photo) on the app's own background. Whatever's selected
/// and on screen is marked with a box; tapping a box frames that face and follows it. Pinch to zoom,
/// drag to look around when zoomed.
struct StageView: View {
    let model: EditorModel
    @State private var pinchFrom: CGFloat?
    @State private var shift: CGSize = .zero
    @State private var dragFrom: CGSize?
    @State private var poofs: [PoofSpot] = []

    private struct PoofSpot: Identifiable {
        let id = UUID()
        let at: CGPoint
        let size: CGFloat
    }

    private var image: UIImage? { model.mode == .photo ? model.photoStageImage : model.stageImage }

    var body: some View {
        GeometryReader { geo in
            let stage = geo.size
            let aspect = model.pictureAspect
            let width = min(stage.width, stage.height * aspect)
            let fit = CGSize(width: width, height: width / aspect)
            let cam = model.camera(fit: fit, stage: stage)
            ZStack(alignment: .topLeading) {
                Color.clear
                if let image {
                    Image(uiImage: image)
                        .resizable()
                        .frame(width: fit.width, height: fit.height)
                        .keyframeAnimator(initialValue: CGFloat(1), trigger: model.photoIndex) { view, p in
                            view.offset(x: model.photoDirection * 22 * (1 - p)).opacity(Double(p))
                        } keyframes: { _ in
                            KeyframeTrack(\.self) {
                                MoveKeyframe(CGFloat(0))
                                CubicKeyframe(CGFloat(1), duration: Motion.move)
                            }
                        }
                        .scaleEffect(cam.z)
                        .offset(x: (0.5 - cam.x) * fit.width * cam.z, y: (0.5 - cam.y) * fit.height * cam.z)
                        .animation(.spring(duration: Motion.move, bounce: 0), value: cam)
                        .scaleEffect(model.focus == nil ? model.zoom : 1)
                        .offset(model.focus == nil ? shift : .zero)
                        .opacity(model.focusAway ? 0.45 : 1)
                        .animation(Motion.ease(Motion.quick), value: model.focusAway)
                        .frame(width: stage.width, height: stage.height)
                }
                boxes(stage: stage, fit: fit)
                if model.before {
                    Text("Before")
                        .font(.ui(12, 600))
                        .foregroundStyle(Ink.keyFg)
                        .padding(.horizontal, 8)
                        .frame(height: 24)
                        .background { cornerShape.fill(Ink.keyBg).background { cornerShape.fill(Ink.keyEdge).offset(y: Metric.edgeSmall) } }
                        .padding(.leading, Metric.gutter)
                        .padding(.top, 8)
                        .transaction { $0.animation = nil }
                }
                ForEach(poofs) { poof in
                    Poof(size: poof.size).position(poof.at)
                }
            }
            .overlay(alignment: .bottom) {
                if model.focus != nil {
                    focusChip.padding(.bottom, 10).transition(NoCatchFade())
                }
            }
            .animation(Motion.ease(Motion.quick), value: model.focus != nil)
            .clipped()
            .contentShape(Rectangle())
            .gesture(pinch)
            .simultaneousGesture(look)
            .onChange(of: model.exportRunning) { _, now in
                if now { withAnimation(Motion.ease(Motion.move)) { shift = .zero } }
            }
            .onChange(of: model.poofTick) { _, _ in
                addPoof(stage: stage, fit: fit, cam: cam)
            }
        }
    }

    private func screen(_ box: CGRect, stage: CGSize, fit: CGSize, cam: EditorModel.Camera) -> CGRect {
        CGRect(x: stage.width / 2 + (box.minX - cam.x) * fit.width * cam.z,
               y: stage.height / 2 + (box.minY - cam.y) * fit.height * cam.z,
               width: box.width * fit.width * cam.z, height: box.height * fit.height * cam.z)
    }

    private func addPoof(stage: CGSize, fit: CGSize, cam: EditorModel.Camera) {
        guard !UIAccessibility.isReduceMotionEnabled else { return }
        for b in model.poofBoxes {
            let box = screen(b, stage: stage, fit: fit, cam: cam)
            let spot = PoofSpot(at: CGPoint(x: box.midX, y: box.midY), size: max(70, box.width * 1.15))
            poofs.append(spot)
            Task { @MainActor in
                try? await Task.sleep(for: .milliseconds(700))
                poofs.removeAll { $0.id == spot.id }
            }
        }
    }

    // Corner brackets in the person's colour, leaving the face itself clear.
    private func boxes(stage: CGSize, fit: CGSize) -> some View {
        let shown = model.focus == nil ? model.visibleBoxes : []
        let k = model.zoom
        return ZStack(alignment: .topLeading) {
            ForEach(shown) { box in
                let w = (box.rect.width * fit.width * k).rounded(), h = (box.rect.height * fit.height * k).rounded()
                let x = (stage.width / 2 + (box.rect.minX - 0.5) * fit.width * k + shift.width).rounded()
                let y = (stage.height / 2 + (box.rect.minY - 0.5) * fit.height * k + shift.height).rounded()
                BoxCorners()
                    .stroke(Ink.faces[box.color % Ink.faces.count], style: StrokeStyle(lineWidth: 2.5, lineCap: .round, lineJoin: .round))
                    .shadow(color: .black.opacity(0.5), radius: 1, y: 1)
                    .frame(width: w, height: h)
                    .contentShape(Rectangle())
                    .onTapGesture { model.enterFocus(box.id) }
                    .position(x: x + w / 2, y: y + h / 2)
                    .transition(.opacity)
                    .accessibilityLabel("Look closer")
                    .accessibilityAddTraits(.isButton)
            }
        }
        .frame(width: stage.width, height: stage.height, alignment: .topLeading)
        .animation(Motion.ease(Motion.quick), value: shown.map(\.id))
    }

    // What the crop is on, and when they're off screen, when they're back.
    private var focusChip: some View {
        let crop = model.focusCrop
        return HStack(spacing: 0) {
            FaceCrop(image: crop.image, box: crop.box, size: 26)
                .padding(.leading, 4)
                .padding(.trailing, 8)
            Group {
                if !model.focusAway {
                    Text(model.focusName)
                } else if let back = model.nextBack {
                    Button { model.seek(back + 0.02) } label: {
                        HStack(spacing: 5) {
                            Text("Off screen · back at \(EditorModel.timecode(back))")
                            IconView(.chevronRight, size: 13, weight: 2.4)
                        }
                        .frame(height: 34)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                } else {
                    Text("Off screen")
                }
            }
            .font(.ui(12, 600))
            .foregroundStyle(Ink.keyFg)
            .lineLimit(1)
            Button { model.exitFocus() } label: {
                IconView(.close, size: 14, weight: 2.2).opacity(0.6)
                    .frame(width: 34, height: 34)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .foregroundStyle(Ink.keyFg)
            .accessibilityLabel("Back to the whole picture")
        }
        .frame(height: 34)
        .background { cornerShape.fill(Ink.keyBg).background { cornerShape.fill(Ink.keyEdge).offset(y: Metric.edgeSmall) } }
    }

    private var pinch: some Gesture {
        MagnifyGesture()
            .onChanged { g in
                guard model.focus == nil else { return }
                if pinchFrom == nil { pinchFrom = model.zoom }
                model.zoom = min(max((pinchFrom ?? 1) * g.magnification, 1), 8)
            }
            .onEnded { _ in
                pinchFrom = nil
                if model.zoom == 1 { withAnimation(Motion.ease(Motion.move)) { shift = .zero } }
            }
    }

    private var look: some Gesture {
        DragGesture(minimumDistance: 6)
            .onChanged { g in
                guard model.zoom > 1, model.focus == nil else { return }
                if dragFrom == nil { dragFrom = shift }
                shift = CGSize(width: (dragFrom?.width ?? 0) + g.translation.width,
                               height: (dragFrom?.height ?? 0) + g.translation.height)
            }
            .onEnded { _ in dragFrom = nil }
    }
}

/// The four corners of a rectangle, softened.
private struct BoxCorners: Shape {
    func path(in full: CGRect) -> Path {
        let r = full.insetBy(dx: 1.25, dy: 1.25)
        let l = min(14, min(r.width, r.height) / 2), c = Metric.corner
        var p = Path()
        func corner(_ a: CGPoint, _ b: CGPoint, _ at: CGPoint, _ d: CGPoint) {
            p.move(to: a)
            p.addLine(to: b)
            p.addQuadCurve(to: d, control: at)
        }
        func line(_ to: CGPoint) { p.addLine(to: to) }
        corner(CGPoint(x: r.minX, y: r.minY + l), CGPoint(x: r.minX, y: r.minY + c), CGPoint(x: r.minX, y: r.minY), CGPoint(x: r.minX + c, y: r.minY))
        line(CGPoint(x: r.minX + l, y: r.minY))
        corner(CGPoint(x: r.maxX - l, y: r.minY), CGPoint(x: r.maxX - c, y: r.minY), CGPoint(x: r.maxX, y: r.minY), CGPoint(x: r.maxX, y: r.minY + c))
        line(CGPoint(x: r.maxX, y: r.minY + l))
        corner(CGPoint(x: r.maxX, y: r.maxY - l), CGPoint(x: r.maxX, y: r.maxY - c), CGPoint(x: r.maxX, y: r.maxY), CGPoint(x: r.maxX - c, y: r.maxY))
        line(CGPoint(x: r.maxX - l, y: r.maxY))
        corner(CGPoint(x: r.minX + l, y: r.maxY), CGPoint(x: r.minX + c, y: r.maxY), CGPoint(x: r.minX, y: r.maxY), CGPoint(x: r.minX, y: r.maxY - c))
        line(CGPoint(x: r.minX, y: r.maxY - l))
        return p
    }
}

// MARK: - Photos

/// For photos: the photos as a thin strip, and the chosen photo's faces as large tiles, each showing
/// the character it becomes. A tap selects a face; characters are given from the Characters tool.
struct PhotoPane: View {
    let model: EditorModel

    private var photo: Photo? {
        model.doc.photos.indices.contains(model.photoIndex) ? model.doc.photos[model.photoIndex] : nil
    }

    var body: some View {
        VStack(spacing: 0) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 4) {
                    ForEach(Array(model.doc.photos.enumerated()), id: \.element.id) { i, p in
                        photoTile(p, on: i == model.photoIndex)
                            .onTapGesture { model.selectPhoto(i) }
                    }
                    Button { model.picking = .morePhotos } label: {
                        IconView(.plus, size: 20)
                            .foregroundStyle(Ink.muted)
                            .frame(width: 44, height: 40)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Add photos")
                }
                .padding(.horizontal, Metric.gutter)
                .padding(.vertical, 2)
            }
            .frame(height: 44)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(alignment: .top, spacing: 8) {
                    ForEach(photo?.faces ?? []) { face in faceTile(face) }
                }
                .padding(.horizontal, Metric.gutter)
                .padding(.top, 3)
            }
            .frame(height: 110)
        }
        .frame(height: Metric.timeline)
        .background(Ink.bg)
    }

    private func photoTile(_ p: Photo, on: Bool) -> some View {
        let aspect = CGFloat(p.width) / CGFloat(max(p.height, 1))
        return Group {
            if let image = model.photoImages[p.id] { Image(uiImage: image).resizable().scaledToFill() } else { Ink.panel2 }
        }
        .frame(width: 40 * min(max(aspect, 0.5), 2.2), height: 40)
        .clipShape(cornerShape)
        .overlay { if !p.found { Pulse() } }
        .overlay { if on { cornerShape.strokeBorder(Ink.ring, lineWidth: 2) } }
        .overlay(alignment: .topLeading) { if on { CheckSeal(size: 12).padding(2).stamped(on: on) } }
    }

    private func faceTile(_ face: PhotoFace) -> some View {
        let c = model.character(face.s.character)
        let on = model.pickedFaces.contains(face.id)
        let crop = model.faceCrop(face)
        return VStack(alignment: .leading, spacing: 5) {
            FaceCrop(image: crop.image, box: crop.box, size: 84)
                .overlay { if on { cornerShape.strokeBorder(Ink.ring, lineWidth: 2) } }
                .overlay(alignment: .bottomTrailing) {
                    if let image = c?.image {
                        Image(uiImage: image).resizable().scaledToFill()
                            .frame(width: 24, height: 32)
                            .clipShape(markShape)
                            .overlay { markShape.strokeBorder(Ink.bg, lineWidth: 1.5) }
                            .padding(4)
                            .transition(.opacity)
                    }
                }
            HStack(spacing: 4) {
                Text(face.label).font(.ui(12, 600)).foregroundStyle(Ink.fg).lineLimit(1)
                if let c { Text(c.name).font(.ui(11, 500)).foregroundStyle(Ink.faint).lineLimit(1) }
            }
            .frame(width: 84, alignment: .leading)
        }
        .contentShape(Rectangle())
        .onTapGesture { model.toggle(face: face.id) }
        .animation(Motion.ease(Motion.quick), value: c?.id)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(c.map { "\(face.label), \($0.name)" } ?? face.label)
        .accessibilityAddTraits(on ? .isSelected : [])
    }
}

/// A small red seal that pulses while something is on its way.
struct Pulse: View {
    @State private var on = false

    var body: some View {
        markShape.fill(Ink.accent)
            .frame(width: 10, height: 10)
            .rotationEffect(.degrees(-6))
            .opacity(on ? 1 : 0.35)
            .onAppear { withAnimation(Motion.ease(Motion.flourish).repeatForever(autoreverses: true)) { on = true } }
            .accessibilityLabel("Finding faces")
    }
}
