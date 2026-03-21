import { API_BASE_URL } from "./config/api.js";

// video INVIZIBIL
const video = document.createElement("video");
video.autoplay = true;
video.playsInline = true;
video.style.display = "none";
document.body.appendChild(video);

// UI
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");

const title = document.getElementById("letter-title");
const image = document.getElementById("tutorial-image");
const statusEl = document.getElementById("status");
const backBtn = document.getElementById("back-btn");
const checkBtn = document.getElementById("check-btn");

const token = localStorage.getItem("firebaseToken");

if (!token) {
    window.location.href = "index.html";
}

// letter din URL
const params = new URLSearchParams(window.location.search);
const targetLetter = params.get("letter");

if (!targetLetter) {
    statusEl.textContent = "Invalid letter";
    throw new Error("Missing letter param");
}

// UI setup
title.textContent = `Letter: ${targetLetter}`;
image.src = `assets/${targetLetter}.jpg`;

// canvas
overlay.width = 640;
overlay.height = 480;

// state
let isCorrectNow = false;
let alreadySaved = false;
let lockStatus = false;

// control mesaj UI
let messageUntil = 0;

// buffers
const buffer = [];
const confidenceBuffer = [];

const CONF_THRESHOLD = 0.8;
const FRAMES = 7;

// camera
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: true
        });
        video.srcObject = stream;
    } catch (e) {
        statusEl.textContent = "Camera error";
        console.error(e);
    }
}

startCamera();

// prediction loop
async function predictFrame() {

    if (video.readyState < 2) {
        requestAnimationFrame(predictFrame);
        return;
    }

    const now = Date.now();

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const c = canvas.getContext("2d");

    // flip input
    c.translate(canvas.width, 0);
    c.scale(-1, 1);
    c.drawImage(video, 0, 0);

    const blob = await new Promise(res =>
        canvas.toBlob(res, "image/jpeg")
    );

    const formData = new FormData();
    formData.append("file", blob);

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${token}`
            },
            body: formData
        });

        const data = await response.json();

        const letter = data.letter || "";
        const confidence = data.confidence || 0;

        if (letter) {
            buffer.push(letter);
            confidenceBuffer.push(confidence);
        }

        if (buffer.length > FRAMES) {
            buffer.shift();
            confidenceBuffer.shift();
        }

        let stable = null;
        let avgConfidence = 0;

        if (buffer.length > 0) {
            const counts = {};
            buffer.forEach(l => counts[l] = (counts[l] || 0) + 1);

            stable = Object.keys(counts)
                .reduce((a, b) => counts[a] > counts[b] ? a : b);

            avgConfidence =
                confidenceBuffer.reduce((a, b) => a + b, 0) / confidenceBuffer.length;
        }

        if (now >= messageUntil && !lockStatus) {

            if (stable === targetLetter && avgConfidence > CONF_THRESHOLD) {
                statusEl.textContent = "✔ Correct!";
                isCorrectNow = true;
            } else {
                statusEl.textContent = "Adjust your hand position";
                isCorrectNow = false;
            }
        }

        // overlay
        const img = new Image();
        img.src = `data:image/jpeg;base64,${data.image}`;

        img.onload = () => {
            ctx.clearRect(0, 0, overlay.width, overlay.height);

            ctx.save();
            ctx.scale(-1, 1);
            ctx.drawImage(img, -overlay.width, 0, overlay.width, overlay.height);
            ctx.restore();
        };

    } catch (e) {
        console.error(e);
        statusEl.textContent = "Prediction error";
    }

    requestAnimationFrame(predictFrame);
}

video.addEventListener("loadeddata", predictFrame);

// SAVE LETTER
checkBtn.addEventListener("click", async () => {

    if (!isCorrectNow) {
        alert("You must perform the correct letter first.");
        return;
    }

    try {

        // dacă deja este salvat
        if (alreadySaved) {
            statusEl.textContent = "✔ Already learned";
            messageUntil = Date.now() + 2500;
            lockStatus = true;

            setTimeout(() => {
                lockStatus = false;
            }, 2500);

            return;
        }

        const response = await fetch(
            `${API_BASE_URL}/users/recognize-letter`,
            {
                method: "POST",
                headers: {
                    "Authorization": `Bearer ${token}`,
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    letter: targetLetter
                })
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Save failed");
        }

        alreadySaved = true;

        // mesaj succes
        statusEl.textContent = "✔ Letter learned!";
        messageUntil = Date.now() + 2500;

        lockStatus = true;

        setTimeout(() => {
            lockStatus = false;
        }, 2500);

    } catch (e) {
        console.error("SAVE ERROR:", e);
        statusEl.textContent = e.message;
    }
});

// back
backBtn.addEventListener("click", () => {
    window.location.href = "learning.html";
});