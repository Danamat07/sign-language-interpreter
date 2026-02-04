import { API_BASE_URL } from "./config/api.js";

const video = document.getElementById("webcam");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const letterEl = document.getElementById("predicted-letter");
const messageEl = document.getElementById("message");
const backBtn = document.getElementById("back-btn");

const token = localStorage.getItem("firebaseToken");
if (!token) {
    window.location.href = "index.html";
}

// Majority voting buffer
const VOTING_FRAMES = 7;
const predictionBuffer = [];

// Set canvas size
overlay.width = 640;
overlay.height = 480;

// Start webcam
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;

        // Oglindim video
        video.style.transform = "scaleX(-1)";
    } catch (e) {
        messageEl.textContent = "Cannot access camera";
        console.error(e);
    }
}

startCamera();

// Function: capture frame, send to backend, update overlay
async function predictFrame() {
    if (video.readyState < 2) {
        requestAnimationFrame(predictFrame);
        return;
    }

    // Temporary canvas for capturing frame
    const tmpCanvas = document.createElement("canvas");
    tmpCanvas.width = video.videoWidth;
    tmpCanvas.height = video.videoHeight;
    const tmpCtx = tmpCanvas.getContext("2d");

    // Oglindim frame înainte de trimis
    tmpCtx.translate(tmpCanvas.width, 0);
    tmpCtx.scale(-1, 1);
    tmpCtx.drawImage(video, 0, 0, tmpCanvas.width, tmpCanvas.height);

    // Convert to blob
    const blob = await new Promise(resolve => tmpCanvas.toBlob(resolve, "image/jpeg"));

    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${token}`
            },
            body: formData
        });

        const data = await response.json();

        let predictedLetter = data.letter || "";

        // Majority voting
        if (predictedLetter) {
            predictionBuffer.push(predictedLetter);
            if (predictionBuffer.length > VOTING_FRAMES) predictionBuffer.shift();
            const counts = {};
            predictionBuffer.forEach(l => counts[l] = (counts[l] || 0) + 1);
            predictedLetter = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
        }

        letterEl.textContent = `Letter: ${predictedLetter}`;

        // Draw annotated frame
        const img = new Image();
        img.src = `data:image/jpeg;base64,${data.image}`;
        img.onload = () => {
            // Clear previous
            ctx.clearRect(0, 0, overlay.width, overlay.height);
            // Flip horizontally ca video
            ctx.save();
            ctx.scale(-1, 1);
            ctx.drawImage(img, -overlay.width, 0, overlay.width, overlay.height);
            ctx.restore();
        };

    } catch (e) {
        console.error(e);
        messageEl.textContent = "Prediction failed";
    }

    requestAnimationFrame(predictFrame);
}

// Start prediction loop
video.addEventListener("loadeddata", () => {
    predictFrame();
});

// Back button
backBtn.addEventListener("click", () => {
    window.location.href = "profile.html";
});
