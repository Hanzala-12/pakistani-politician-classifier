# I Built a Face Recognizer for 16 Pakistani Politicians — Here's Everything That Went Wrong (and Right)

*How 3,870 scraped photos became a 96%-accurate classifier, an ArcFace training loop, a React app, and a full MLOps pipeline — plus the parts I'd do differently.*

---

> **TL;DR** — I trained three models to identify 16 Pakistani politicians from a single photo. The best one hits **96.32% test accuracy**. The interesting part isn't the number; it's that ~56% of my scraped images were thrown away before training, why I switched from a normal softmax classifier to **ArcFace**, and how a tiny 136-image test set makes every accuracy figure a little bit of a lie. Code: [github.com/Hanzala-12/pakistani-politician-classifier](https://github.com/Hanzala-12/pakistani-politician-classifier)

**[ FIGURE 1 — hero image · file: `blog/app-casia-batch.jpg` · caption: "The finished app: drop in a folder of photos, get an identity + confidence per image, export CSV." ]**

---

## Why this project

Face recognition tutorials almost always use the same three datasets — LFW, CelebA, VGGFace2 — and the same clean, frontal, well-lit crops. I wanted to know what happens when you skip all of that and build a recognizer for a **specific, messy, real-world set of identities** using nothing but images you can scrape off the open web.

Pakistani politicians turned out to be a good stress test:

- Some (Imran Khan, Nawaz Sharif, Bilawal Bhutto) have **thousands** of high-quality press photos.
- Others (Ahmed Sharif Chaudhry, Barrister Gohar, Asfandyar Wali) have **a few dozen** usable images, often low-resolution, often in group shots.
- Several look genuinely similar in the wrong lighting — older men in glasses, similar beards, similar sherwanis.

That imbalance and ambiguity is the whole point. Here's how it went.

---

## Step 1: Scraping, and immediately regretting it

I collected images with `icrawler`, hitting Bing, Google and DuckDuckGo image search with two hand-written queries per person — e.g. `"Imran Khan Pakistan PM face"` and `"Imran Khan PTI"`.

```python
POLITICIANS = {
    "imran_khan": ["Imran Khan Pakistan PM face", "Imran Khan PTI"],
    "nawaz_sharif": ["Nawaz Sharif Pakistan PM", "Nawaz Sharif PML-N"],
    "asif_ali_zardari": ["Asif Ali Zardari Pakistan President", "Zardari PPP"],
    # ... 13 more
}
```

Two crawl passes across Bing, Google and DuckDuckGo gave me **3,870 raw images** (1,583 from the first pass, 2,287 from a later top-up run).

They were *filthy*. Political image search returns:

- Group photos at press conferences (which face is the label?)
- Rally crowds where the subject is 40 pixels tall
- Newspaper front pages, campaign posters with text overlays
- Cartoons and caricatures
- Plain wrong people (Google is confident that a lot of men in suits are Nawaz Sharif)

A raw scrape is not a dataset. It's a to-do list.

---

## Step 2: MTCNN alignment — the step that deleted half my data

Instead of a loose "does this image contain a face?" filter, I went strict. Every image goes through **MTCNN** (via `facenet-pytorch`), and I keep it **only if exactly one face is detected**. Then I use the eye landmarks to rotate the face level, crop a square around it with a 20% margin, and resize to 336×336.

```python
boxes, probs, landmarks = mtcnn.detect(img_np, landmarks=True)

if boxes is None or landmarks is None:
    removed += 1;  continue

# Keep only single-face images; skip group photos and no-face cases.
if len(boxes) != 1:
    removed += 1;  continue

aligned = align_face_with_landmarks(
    img, boxes[0], landmarks[0], image_size=336, margin_ratio=0.2
)
```

The eye-landmark alignment is worth doing properly. You compute the angle between the eyes, rotate the *whole image* around the eye midpoint, then re-project the bounding box corners through the same rotation so your crop still lands on the face:

```python
dx = right_eye[0] - left_eye[0]
dy = right_eye[1] - left_eye[1]
angle = math.degrees(math.atan2(dy, dx))
eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
rotated = img.rotate(-angle, resample=Image.BICUBIC, center=eye_center, expand=False)
```

Then I ran perceptual-hash (pHash) deduplication per class — the same wire-service photo shows up on 30 news sites, and near-duplicates that straddle your train/test split will inflate your accuracy and lie to you.

**The damage:**

| Stage | Images |
|---|---|
| Raw scrape (Bing + Google + DuckDuckGo) | 3,870 |
| After MTCNN single-face alignment + pHash dedup | **1,687** |
| Split: 75 / 15 / 10, stratified, seed 42 | test set = **136 images**, 4–19 per class |

I threw away **56%** of everything I collected. Group photos and duplicates are ruthless. If you're planning a project like this, scrape 2–3× more than you think you need, then do it again for your smallest classes.

---

## Step 3: Offline augmentation, but only where it's needed

For classes that still had too few images after alignment, I generated 3 augmented copies per image (horizontal flip, ±25° rotation, brightness/contrast jitter) — and **skipped** any class that already had 120+ real images, so I wasn't drowning Imran Khan in synthetic data he didn't need.

```python
aug_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=25, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
])
```

Post-augmentation, each politician had roughly 140–290 training images. Still small. This is a "transfer learning or bust" regime.

---

## Step 4: Why softmax wasn't good enough → ArcFace

My first notebook was a plain ResNet with a 16-way softmax head. It topped out around **76% test accuracy** and made *confident* mistakes — high softmax probability on the wrong person.

The problem with vanilla softmax classification for faces: it only learns boundaries good enough to *separate* the training identities. It has no incentive to make each identity a **tight, well-margined cluster** in feature space. With 16 classes and ~150 images each, "good enough to separate" overfits fast.

**ArcFace** (Additive Angular Margin Loss) fixes the objective. You L2-normalize both the face embedding and the class weight vectors, so classification becomes about the **angle** between them. Then during training you add an angular margin `m` to the true class before the softmax, forcing the network to pull same-identity embeddings meaningfully closer together than it would otherwise bother to.

```python
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_normal_(self.weight)
        self.s, self.m = s, m

    def forward(self, embeddings, labels=None):
        embeddings = F.normalize(embeddings, dim=1)
        W_norm = F.normalize(self.weight, dim=1)
        cosine = embeddings @ W_norm.t()

        if labels is None:              # validation / inference: NO margin
            return self.s * cosine

        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        one_hot = F.one_hot(labels, num_classes=self.weight.size(0)).bool()
        theta_m = torch.where(one_hot, theta + self.m, theta)   # margin on true class only
        return self.s * torch.cos(theta_m)
```

Two details that matter and are easy to get wrong:

1. **The margin is applied at training time only.** Validation and inference call the same module with `labels=None` and get plain scaled cosine logits. If you accidentally leave the margin on during eval, your accuracy looks *worse* and you'll chase a bug that isn't there.
2. **There is no classifier head to save.** The trained artifact is the backbone plus the ArcFace weight matrix `W`. At inference you rebuild logits yourself: `scale * normalize(embedding) @ normalize(W).T`. My checkpoints store `W`, `scale`, and `margin` in an `arcface_eval` dict alongside the class names.

For the backbone I used **InceptionResnetV1** from `facenet-pytorch`, pretrained on faces — and I trained it twice, once from **VGGFace2** weights and once from **CASIA-WebFace** weights, to compare. A plain **ResNet-50** (ImageNet transfer, ordinary CrossEntropy) stayed in as a baseline.

---

## Step 5: The training loop, and its two epoch-6 tricks

Config that mattered:

| Hyperparameter | Value |
|---|---|
| Image size | 336 × 336 |
| Batch size | 32 |
| Optimizer | AdamW — `1e-4` head, `3e-6` backbone |
| ArcFace scale `s` / margin `m` | 64.0 / 0.3 |
| Label smoothing | 0.1 |
| MixUp | α = 0.2, applied 50% of batches |
| Early stopping | patience 10 |
| Test-time augmentation | 5-crop + horizontal flip = 10 views |

**Trick 1 — delayed backbone unfreeze.** For the first 5 epochs I train the head and ArcFace weights while the pretrained backbone stays effectively frozen (a `3e-6` LR is barely a nudge). At **epoch 6** I rebuild the optimizer so the backbone starts actually moving:

```python
if epoch == 6:
    optimizer = torch.optim.AdamW([
        {'params': model_params,   'lr': cfg.BACKBONE_UNFREEZE_LR},  # 3e-6
        {'params': arcface_params, 'lr': cfg.HEAD_LR},               # 1e-4
    ], weight_decay=cfg.WEIGHT_DECAY)
```

The ResNet-50 baseline does the analogous thing — it unfreezes only `fc` + `layer4` at epoch 6 with discriminative learning rates. Letting a big backbone loose on ~1,400 images from step 0 just memorizes them.

**Trick 2 — test-time augmentation.** At evaluation, each test image becomes 10 views (four corners + center crop, each also flipped). The model scores all 10 and I average the logits. It's a cheap ~0.5–1% bump and it makes the predictions less twitchy on off-center faces.

```python
transforms.Compose([
    transforms.Resize(config.IMAGE_SIZE),
    transforms.FiveCrop(config.IMAGE_SIZE),
    transforms.Lambda(lambda crops: tta_stack(crops)),   # -> (10, C, H, W)
])
```

I also hit the classic bug where `class_weights` sits on the CPU while the model is on the GPU and the first forward pass explodes — moved to the correct device *before* building the criterion, which fixed it for every weighted-loss model, not just the one where I noticed it.

**[ FIGURE 2 · file: `blog/curves-casia.jpg` · caption: "CASIA-WebFace backbone. Train loss keeps dropping; val accuracy plateaus near 93% by epoch 8, then early stopping kicks in. Test accuracy with TTA lands higher, at 96.3%." ]**

---

## Results

Evaluated on a held-out test set of **136 images** (4–19 per class), seed 42, fully reproducible:

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| InceptionResnetV1 + ArcFace (**CASIA-WebFace**) | **96.32%** | 0.9645 | 0.9608 | 0.9607 |
| InceptionResnetV1 + ArcFace (VGGFace2) | 95.59% | 0.9689 | 0.9504 | 0.9573 |
| ResNet-50 (ImageNet transfer, CrossEntropy) | 94.85% | 0.9596 | 0.9534 | 0.9550 |

From a 76% baseline to 96% — most of that gain is **data quality** (strict alignment, dedup) and **transfer from face-pretrained weights**, with ArcFace and TTA on top.

The confusion matrix for the best model is almost clean diagonal. The few errors are the plausible ones:

- **Asfandyar Wali ↔ Altaf Hussain** — older men, glasses, similar face shape.
- **Ahmed Sharif Chaudhry ↔ Imran Khan** — a single mixed-up pair, small sample.
- **Nawaz Sharif ↔ Altaf Hussain** — again, glasses and build.

**[ FIGURE 3 · file: `blog/confusion-casia.jpg` · caption: "16×16 confusion matrix, CASIA model, 136 test images. Nearly clean diagonal; every off-diagonal cell is a single image between two visually similar identities." ]**

---

## A thing the demo made obvious: ArcFace models are wildly overconfident

Run the same 7 images through all three models in the app and the confidence column tells a story:

- **ResNet-50** (plain CrossEntropy): 29%, 43%, 67%, 71%, 87%, 94%, 97% — a normal spread.
- **ArcFace (VGGFace2)**: 100%, 100%, 100%, 100%, 100%, 100%, 100%.
- **ArcFace (CASIA)**: 99.9%, 100%, 100%, 100%, 100%, 100%, 100%.

This isn't the ArcFace models being "more sure" in any meaningful sense. Inference logits are `s · cos θ` with **s = 64**, so even a mediocre cosine similarity of 0.6 becomes a logit of ~38, and softmax over logits that large saturates to 1.0 almost regardless of the runner-up. The scale factor that makes ArcFace *train* well also makes its output probabilities **uninformative as confidence**. If you need a real "how sure are we" signal from an ArcFace model, you get it from the raw cosine similarity to the class prototype (and a threshold), not from softmax.

**[ FIGURE 4 — two screenshots, stacked or side-by-side · files: `blog/app-resnet50-batch.jpg` and `blog/app-vggface2-batch.jpg` · caption: "Same input images, two models. ResNet-50 (CrossEntropy) gives a real confidence spread; the ArcFace model reports 100% on everything." ]**

---

## The part nobody blogs about: the test set is tiny

That 96.32% is **131 out of 136**. Change *one* prediction and overall accuracy moves by 0.74 points. Several classes have **4–6 test images** — a per-class F1 for those is essentially a coin-flip estimator. Overall accuracy is stable enough to trust directionally; per-class numbers are *indicative*, not definitive, and I say so in the README.

If I were shipping this for real, the single highest-value thing I could do is **hand-label a proper 500+ image test set** with controlled variety (era, lighting, angle). More than any architecture change.

---

## Wrapping it in something you can actually use

A model in a notebook isn't a project, so I built the rest:

- **Flask inference API** (`/predict`, `/predict/batch`, `/health`) — accepts an upload, runs MTCNN alignment, returns top-3 with confidences. There's also a **FastAPI** version.
- **React 19 + Vite + Tailwind** front end — drag-and-drop single or batch classification, a model picker, CSV export of batch results, glassmorphism because why not.
- **Docker** image (CUDA base, weights volume-mounted, not baked in).
- **MLOps wiring**: DVC for data/model versioning, MLflow for experiment tracking, an Airflow DAG for the collect→train→evaluate pipeline, and a GitHub Actions workflow that tests, builds the image, pushes to Docker Hub, and SSH-deploys to EC2.

Is the MLOps layer overkill for a 1,687-image dataset? Absolutely. It was also the point — I wanted the repo to look like something you'd inherit at work, not a single `.ipynb`.

---

## What I'd do differently

1. **Scrape wider, then scrape again for the tail.** Plan for a 50%+ cull from alignment. Budget queries per-class based on how obscure the person is.
2. **Build the real test set first.** Freeze it, hand-check every label, never augment it.
3. **Try an open-set / verification framing.** Right now the model *must* pick one of 16. A real system needs "none of the above" — which ArcFace embeddings are actually well-suited to, via a cosine-similarity threshold against enrolled prototypes.
4. **Log embeddings, not just metrics.** A UMAP of the 512-d space per epoch would have told me what ArcFace was doing far faster than the loss curve did.
5. **Drop one backbone.** VGGFace2 and CASIA within a point of each other didn't justify the training time; I'd keep CASIA and spend the compute on data.

---

## Responsible use — not a footnote

This is facial recognition trained on images of **real people**, scraped from the public web **without consent**, for **education and research only**. It is not built for surveillance, profiling, or any automated decision that affects a person, and the training data almost certainly carries collection bias. If you fork it, that responsibility comes with you. The repo has a full Responsible Use section; please actually read it.

---

## Links

- **Code:** [github.com/Hanzala-12/pakistani-politician-classifier](https://github.com/Hanzala-12/pakistani-politician-classifier)
- **Stack:** PyTorch 2.0 · facenet-pytorch (MTCNN + InceptionResnetV1) · Albumentations · Flask / FastAPI · React 19 + Vite · Docker · DVC + MLflow + Airflow + GitHub Actions

*If you've built something similar for a different identity set, I'd genuinely like to hear how your alignment cull rate compared to mine — 56% still feels high.*
