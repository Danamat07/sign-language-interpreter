const params = new URLSearchParams(window.location.search);
const letter = params.get("letter");

const title = document.getElementById("title");
const video = document.getElementById("video");
const desc = document.getElementById("description");
const backBtn = document.getElementById("back-btn");

if (!letter) {
    title.textContent = "Error";
    throw new Error("No letter");
}

// UI
title.textContent = `Letter ${letter}`;
video.src = `assets/${letter}.mp4`;

// TEXT EXPLICATIV
if (letter === "J") {
    desc.textContent = "The letter J is dynamic. Move your hand in a 'J' shape in the air.";
}

if (letter === "Z") {
    desc.textContent = "The letter Z is dynamic. Trace a 'Z' shape using your index finger.";
}

// back
backBtn.addEventListener("click", () => {
    window.location.href = "learning.html";
});