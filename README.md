# FaceJutsu

**Face swap no jutsu.** Swap faces in your photos and videos with one click, right on your own computer. Fast, lightweight and completely private: your files never leave your machine.

**[Try it in your browser](https://i2m3d.github.io/FaceJutsu/)**: nothing to install.

![Selecting a face, giving it a character, and the swap](assets/demo.gif)

## What it does

- **Photos and videos:** swap one face or everyone in the shot, and keep the original audio.
- **Characters:** make a character from a photo and reuse it whenever you like.
- **A real editor:** each person gets a track on the timeline, so you can split, trim and change who appears when.
- **Live preview:** see the result as you edit, and the export reuses what the preview already made.
- **Hands and glasses stay in front:** objects covering a face can stay on top of the swap.
- **Private by design:** everything runs on your own graphics card. There are no accounts and nothing is uploaded.

![The FaceJutsu editor after a swap](assets/swapped.png)

## Get started

The easiest way is in your browser, with nothing to install:

1. Open **[FaceJutsu](https://i2m3d.github.io/FaceJutsu/)** in a recent version of **Chrome or Edge**, on a computer with a graphics card.
2. Choose **Download all** to fetch the models. They stay in your browser for next time.
3. Start swapping. Everything runs on your own graphics card, and you can watch your browser's network tab to see that your files aren't uploaded.

![The start page, ready to download the models](assets/start.png)

### Or download the desktop app

The desktop app is optional. It's the same app, but it keeps your files in folders next to it instead of in your browser.

**The downloads aren't code-signed**, so Windows and macOS warn you before running them. That's a reasonable warning for an app from a publisher they can't verify. If you'd rather not run it, use the browser version instead.

1. Download the zip for your system from the [latest release](../../releases/latest), and unzip it somewhere with room to spare.
2. Start FaceJutsu:
   - **Windows** (`FaceJutsu-windows-x64.zip`): open `FaceJutsu.exe`. Windows warns about apps it doesn't recognize. To run it anyway, choose **More info**, then **Run anyway**. The first launch can take a moment while Windows checks it.
   - **macOS** (`FaceJutsu-macos-arm64.zip` for Apple silicon, `FaceJutsu-macos-x64.zip` for Intel): macOS blocks apps from outside the App Store. To run it anyway, open Terminal in the unzipped folder and run `xattr -dr com.apple.quarantine .`, which removes the "downloaded from the internet" mark from that folder. Then run `./FaceJutsu`.
   - **Linux** (`FaceJutsu-linux-x64.zip`): in a terminal, in the unzipped folder, run `./FaceJutsu`. Chrome may need WebGPU turned on at `chrome://flags`.
3. The app opens in your browser. Choose **Download all** to fetch the models, then start swapping.

You'll need a recent version of **Chrome or Edge** and a computer with a graphics card. Keep the FaceJutsu window open while you work.

## Your files

**In the browser**, your models and saved work stay in the browser's private storage for the site, so clearing the site's data removes them. Finished photos and videos arrive as normal downloads, or with **Save as…**, a video goes straight to a file you pick.

**In the desktop app**, everything stays in the folder next to the app:

| Folder or file | What's in it |
| --- | --- |
| `outputs` | your finished photos and videos |
| `models` | the model files |
| `data` | your characters, your edits and swapped frames kept to save time |
| `config.toml` | settings, such as the port and where the models come from |

## Models

FaceJutsu doesn't come with any models. The start page lists what's needed and downloads it for you. In the desktop app, if a link ever stops working, put the file in the `models` folder yourself, or change its address in `config.toml`.

| File | What it does | |
| --- | --- | --- |
| `scrfd-10g.bf16.safetensors` | finds the faces | needed |
| `arcface-r50.int8.safetensors` | recognizes each person | needed |
| `inswapper-128.int8.safetensors` | swaps the face | needed |
| `gpen-bfr-256.int8.safetensors` | sharpens faces, fast | optional |
| `gpen-bfr-512.int8.safetensors` | sharpens faces, best | optional |
| `xseg.int8.safetensors` | keeps hands and glasses in front | optional |

Each model has its own license, which is yours to check before using it or what you make with it. Some are for non-commercial research only, and one states no license:

- **inswapper_128, SCRFD, ArcFace:** [InsightFace](https://github.com/deepinsight/insightface), non-commercial research license
- **Face enhancer:** [GPEN](https://github.com/yangxy/GPEN) (Yang et al.), no stated license
- **Occlusion mask:** community-trained XSeg, on the architecture of [DeepFaceLab](https://github.com/iperov/DeepFaceLab) (GPL-3.0)

## Use it responsibly

Only swap the faces of people who have agreed to it. Don't use FaceJutsu to deceive, harass or impersonate anyone, or to make sexual images of anyone without their consent, or of minors at all. What you make and share is your responsibility.

## License

FaceJutsu is provided as is, without warranty of any kind. The open source software it's built with is listed in the app, under **Storage & about**, then **Open source licenses**.
