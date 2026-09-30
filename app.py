import time
import numpy as np
import torch
import torch.nn as nn
import gradio as gr

from PIL import Image
from torchvision import transforms
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
)


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 3
IMG_WIDTH = 640
IMG_HEIGHT = 384

UNET_PATH = "models/unet_3class_v1.pth"
SEGFORMER_PATH = "models/segformer_b0_carla_best.pth"

SEGFORMER_CHECKPOINT = "nvidia/segformer-b0-finetuned-ade-512-512"

ID2LABEL = {
    0: "background",
    1: "road",
    2: "lane",
}

LABEL2ID = {
    "background": 0,
    "road": 1,
    "lane": 2,
}


# Visualization colors
# background = black
# road       = gray
# lane       = white
CLASS_COLORS = np.array(
    [
        [0, 0, 0],
        [128, 128, 128],
        [255, 255, 255],
    ],
    dtype=np.uint8,
)


# ============================================================
# U-NET
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetMultiClass(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.d1 = DoubleConv(n_channels, 32)
        self.p1 = nn.MaxPool2d(2)

        self.d2 = DoubleConv(32, 64)
        self.p2 = nn.MaxPool2d(2)

        self.d3 = DoubleConv(64, 128)
        self.p3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up3 = nn.ConvTranspose2d(
            256, 128, 2, stride=2
        )
        self.u3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(
            128, 64, 2, stride=2
        )
        self.u2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(
            64, 32, 2, stride=2
        )
        self.u1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(
            32,
            n_classes,
            kernel_size=1,
        )

    def forward(self, x):

        c1 = self.d1(x)
        p1 = self.p1(c1)

        c2 = self.d2(p1)
        p2 = self.p2(c2)

        c3 = self.d3(p2)
        p3 = self.p3(c3)

        bn = self.bottleneck(p3)

        up_3 = self.up3(bn)
        merge3 = torch.cat([up_3, c3], dim=1)
        x = self.u3(merge3)

        up_2 = self.up2(x)
        merge2 = torch.cat([up_2, c2], dim=1)
        x = self.u2(merge2)

        up_1 = self.up1(x)
        merge1 = torch.cat([up_1, c1], dim=1)
        x = self.u1(merge1)

        return self.out(x)


# ============================================================
# MODEL LOADING
# ============================================================

print(f"Using device: {DEVICE}")


# ----------------------------
# U-Net
# ----------------------------

unet = UNetMultiClass(
    n_channels=3,
    n_classes=NUM_CLASSES,
)

unet_checkpoint = torch.load(
    UNET_PATH,
    map_location=DEVICE,
)

if isinstance(unet_checkpoint, dict) and "model_state_dict" in unet_checkpoint:
    unet.load_state_dict(
        unet_checkpoint["model_state_dict"]
    )
elif isinstance(unet_checkpoint, dict) and "state_dict" in unet_checkpoint:
    unet.load_state_dict(
        unet_checkpoint["state_dict"]
    )
else:
    unet.load_state_dict(unet_checkpoint)

unet.to(DEVICE)
unet.eval()


# ----------------------------
# SegFormer
# ----------------------------

segformer_processor = AutoImageProcessor.from_pretrained(
    SEGFORMER_CHECKPOINT
)

segformer = SegformerForSemanticSegmentation.from_pretrained(
    SEGFORMER_CHECKPOINT,
    num_labels=NUM_CLASSES,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)

segformer_checkpoint = torch.load(
    SEGFORMER_PATH,
    map_location=DEVICE,
)

if (
    isinstance(segformer_checkpoint, dict)
    and "model_state_dict" in segformer_checkpoint
):
    segformer.load_state_dict(
        segformer_checkpoint["model_state_dict"]
    )
else:
    segformer.load_state_dict(
        segformer_checkpoint
    )

segformer.to(DEVICE)
segformer.eval()


# ============================================================
# PREPROCESSING
# ============================================================

unet_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMG_HEIGHT, IMG_WIDTH)
        ),
        transforms.ToTensor(),
    ]
)


# ============================================================
# WARM-UP
# ============================================================

def warmup_models():
    """
    Warm up both models once so that first-call CUDA/kernel
    initialization does not contaminate user-facing latency.
    """

    try:

        dummy_image = Image.new(
            "RGB",
            (IMG_WIDTH, IMG_HEIGHT),
        )

        # U-Net warmup
        dummy_unet = (
            unet_transform(dummy_image)
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():
            for _ in range(3):
                _ = unet(dummy_unet)

        # SegFormer warmup
        dummy_segformer = (
            segformer_processor(
                dummy_image,
                return_tensors="pt",
            )
        )

        dummy_pixel_values = (
            dummy_segformer["pixel_values"]
            .to(DEVICE)
        )

        with torch.no_grad():
            for _ in range(3):
                _ = segformer(
                    pixel_values=dummy_pixel_values
                )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        print("Model warm-up completed.")

    except Exception as e:
        print(f"Warm-up warning: {e}")


warmup_models()


# ============================================================
# UTILITIES
# ============================================================

def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def create_mask(prediction):
    """
    Convert class prediction [H, W] into RGB mask.
    """

    prediction = prediction.astype(np.uint8)

    mask = CLASS_COLORS[prediction]

    return mask


def create_overlay(image, mask):
    """
    Create segmentation overlay.
    """

    image_np = np.asarray(image.convert("RGB"))

    mask_image = Image.fromarray(mask).resize(
        image.size,
        Image.Resampling.NEAREST,
    )

    mask_np = np.asarray(mask_image)

    overlay = (
        image_np.astype(np.float32) * 0.65
        + mask_np.astype(np.float32) * 0.35
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)

    return overlay


def make_empty_output():
    return None


# ============================================================
# U-NET INFERENCE
# ============================================================

def run_unet(image):
    """
    Run U-Net inference.

    Timing includes only the model forward pass.
    """

    if image is None:
        raise ValueError("Please upload an image first.")

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")

    tensor = (
        unet_transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        synchronize()

        start = time.perf_counter()

        logits = unet(tensor)

        synchronize()

        end = time.perf_counter()

    latency_ms = (
        end - start
    ) * 1000.0

    prediction = (
        torch.argmax(
            logits,
            dim=1,
        )
        .squeeze(0)
        .cpu()
        .numpy()
    )

    mask = create_mask(prediction)

    overlay = create_overlay(
        image,
        mask,
    )

    fps = (
        1000.0 / latency_ms
        if latency_ms > 0
        else 0.0
    )

    return (
        mask,
        overlay,
        latency_ms,
        fps,
    )


# ============================================================
# SEGFORMER INFERENCE
# ============================================================

def run_segformer(image):
    """
    Run SegFormer-B0 inference.

    Timing includes only the model forward pass.
    """

    if image is None:
        raise ValueError("Please upload an image first.")

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")

    inputs = segformer_processor(
        image,
        return_tensors="pt",
    )

    pixel_values = (
        inputs["pixel_values"]
        .to(DEVICE)
    )

    with torch.no_grad():

        synchronize()

        start = time.perf_counter()

        outputs = segformer(
            pixel_values=pixel_values
        )

        synchronize()

        end = time.perf_counter()

    latency_ms = (
        end - start
    ) * 1000.0

    logits = outputs.logits

    # Resize logits to the target CARLA resolution
    logits = torch.nn.functional.interpolate(
        logits,
        size=(IMG_HEIGHT, IMG_WIDTH),
        mode="bilinear",
        align_corners=False,
    )

    prediction = (
        torch.argmax(
            logits,
            dim=1,
        )
        .squeeze(0)
        .cpu()
        .numpy()
    )

    mask = create_mask(prediction)

    overlay = create_overlay(
        image,
        mask,
    )

    fps = (
        1000.0 / latency_ms
        if latency_ms > 0
        else 0.0
    )

    return (
        mask,
        overlay,
        latency_ms,
        fps,
    )


# ============================================================
# MAIN INFERENCE ROUTER
# ============================================================

def run_inference(image, model_choice):

    if image is None:
        raise gr.Error(
            "Please upload a CARLA image first."
        )

    if model_choice == "U-Net":

        (
            unet_mask,
            unet_overlay,
            unet_latency,
            unet_fps,
        ) = run_unet(image)

        return (
            unet_mask,
            unet_overlay,
            f"{unet_latency:.3f} ms",
            f"{unet_fps:.2f} FPS",

            None,
            None,
            "—",
            "—",
        )

    elif model_choice == "SegFormer-B0":

        (
            seg_mask,
            seg_overlay,
            seg_latency,
            seg_fps,
        ) = run_segformer(image)

        return (
            None,
            None,
            "—",
            "—",

            seg_mask,
            seg_overlay,
            f"{seg_latency:.3f} ms",
            f"{seg_fps:.2f} FPS",
        )

    elif model_choice == "Run Both":

        # Sequential inference:
        # U-Net first, SegFormer second.

        (
            unet_mask,
            unet_overlay,
            unet_latency,
            unet_fps,
        ) = run_unet(image)

        (
            seg_mask,
            seg_overlay,
            seg_latency,
            seg_fps,
        ) = run_segformer(image)

        return (
            unet_mask,
            unet_overlay,
            f"{unet_latency:.3f} ms",
            f"{unet_fps:.2f} FPS",

            seg_mask,
            seg_overlay,
            f"{seg_latency:.3f} ms",
            f"{seg_fps:.2f} FPS",
        )

    raise gr.Error("Unknown model selection.")


# ============================================================
# CSS
# ============================================================

CSS = """
:root {
    --card-radius: 16px;
}

.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
}

.hero {
    padding: 28px 30px;
    border-radius: 18px;
    margin-bottom: 20px;
    border: 1px solid rgba(128,128,128,0.25);
}

.hero h1 {
    margin-bottom: 8px;
    font-size: 32px;
}

.hero p {
    margin: 4px 0;
    opacity: 0.85;
}

.model-card {
    border-radius: 16px !important;
}

.metric-box {
    text-align: center;
}

.metric-box input {
    text-align: center !important;
    font-weight: 600 !important;
    font-size: 18px !important;
}

.section-title {
    margin-top: 10px;
    margin-bottom: 8px;
}

.footer {
    text-align: center;
    opacity: 0.65;
    margin-top: 20px;
    font-size: 13px;
}
"""


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="CARLA Semantic Segmentation",
    css=CSS,
    theme=gr.themes.Soft(),
) as demo:

    gr.HTML(
        """
        <div class="hero">
            <h1>CARLA Semantic Segmentation</h1>

            <p>
                <b>U-Net vs SegFormer-B0</b>
                &nbsp;·&nbsp;
                Autonomous Driving Computer Vision
            </p>

            <p>
                Upload a CARLA image and compare
                pixel-level road and lane segmentation.
            </p>
        </div>
        """
    )

    with gr.Row():

        with gr.Column(
            scale=1,
            variant="panel",
        ):

            gr.Markdown(
                "### Input",
                elem_classes="section-title",
            )

            input_image = gr.Image(
                label="CARLA RGB Image",
                type="pil",
                image_mode="RGB",
                height=300,
            )

            model_choice = gr.Radio(
                choices=[
                    "U-Net",
                    "SegFormer-B0",
                    "Run Both",
                ],
                value="Run Both",
                label="Inference Model",
            )

            run_button = gr.Button(
                "Run Inference",
                variant="primary",
                size="lg",
            )

            gr.Markdown(
                """
                **Classes**

                ⬛ Background &nbsp;&nbsp;
                ◼️ Road &nbsp;&nbsp;
                ⬜ Lane

                **Latency:** pure model forward pass.
                Preprocessing and visualization are excluded.
                """,
            )

        with gr.Column(
            scale=2,
        ):

            gr.Markdown(
                "### U-Net",
                elem_classes="section-title",
            )

            with gr.Row():

                unet_mask = gr.Image(
                    label="Segmentation Mask",
                    type="numpy",
                    height=280,
                )

                unet_overlay = gr.Image(
                    label="Overlay",
                    type="numpy",
                    height=280,
                )

            with gr.Row():

                unet_latency = gr.Textbox(
                    label="Pure Inference Latency",
                    value="—",
                    interactive=False,
                    elem_classes="metric-box",
                )

                unet_fps = gr.Textbox(
                    label="Corresponding FPS",
                    value="—",
                    interactive=False,
                    elem_classes="metric-box",
                )

            gr.Markdown(
                "---"
            )

            gr.Markdown(
                "### SegFormer-B0",
                elem_classes="section-title",
            )

            with gr.Row():

                seg_mask = gr.Image(
                    label="Segmentation Mask",
                    type="numpy",
                    height=280,
                )

                seg_overlay = gr.Image(
                    label="Overlay",
                    type="numpy",
                    height=280,
                )

            with gr.Row():

                seg_latency = gr.Textbox(
                    label="Pure Inference Latency",
                    value="—",
                    interactive=False,
                    elem_classes="metric-box",
                )

                seg_fps = gr.Textbox(
                    label="Corresponding FPS",
                    value="—",
                    interactive=False,
                    elem_classes="metric-box",
                )

    gr.HTML(
        """
        <div class="footer">
            CARLA Semantic Segmentation Benchmark
            · U-Net vs SegFormer-B0
            · 3-class semantic segmentation
        </div>
        """
    )


    run_button.click(
        fn=run_inference,
        inputs=[
            input_image,
            model_choice,
        ],
        outputs=[
            unet_mask,
            unet_overlay,
            unet_latency,
            unet_fps,

            seg_mask,
            seg_overlay,
            seg_latency,
            seg_fps,
        ],
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.launch()