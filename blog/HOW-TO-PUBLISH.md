# Medium draft — status

**The draft is already built on your Medium account** (it's open in the Chrome tab
Claude was driving):

<https://medium.com/p/dd189bf68a16/edit>

It has the full article: title, subtitle, TL;DR, all section headings, code blocks,
bullet/numbered lists, the three data tables rewritten as prose, and all five images
in place with captions:

| Figure | Image | Placement |
|---|---|---|
| Hero | `app-casia-batch.jpg` | under the TL;DR |
| Training curves | `curves-casia.jpg` | end of "Step 5: The training loop" |
| Confusion matrix | `confusion-casia.jpg` | end of "Results" |
| Two app screenshots (side by side) | `app-resnet50-batch.jpg` + `app-vggface2-batch.jpg` | "ArcFace models are wildly overconfident" |

## Before you publish — quick review

1. **Read it through once** in the editor. Everything below is optional polish.
2. **Code-block language labels.** Medium auto-guessed the language for each code block
   ("Auto (Makefile)", "Auto (Ruby)", …). To fix: click the little language label above
   a block → choose **Python**. Cosmetic; the code renders fine either way.
3. **The divider right under the subtitle** (`· · ·`) is optional — click it and press
   Backspace if you'd rather it wasn't there.
4. **Tags** (Publish dialog, max 5): `machine-learning`, `deep-learning`, `pytorch`,
   `computer-vision`, `face-recognition`.
5. **Featured image / social preview**: Medium will use the hero screenshot by default.

## Publish

Click **Publish** (top right) → set tags → **Publish now**. Claude deliberately did not
click Publish — a public post under your name is yours to send.

## Re-generating from scratch

The source of truth is `pakistani-politician-classifier-medium.md` in this folder.
If you want to redo the import: `gh gist create pakistani-politician-classifier-medium.md`,
then <https://medium.com/p/import> with that URL, then re-add the five images.
