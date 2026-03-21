import { API_BASE_URL } from "./config/api.js";

const grid = document.getElementById("letters-grid");
const progressText = document.getElementById("progress-text");
const backBtn = document.getElementById("back-btn");

const token = localStorage.getItem("firebaseToken");

if (!token) {
    window.location.href = "index.html";
}

// alfabet
const ALPHABET = [
    "A","B","C","D","E",
    "F","G","H","I",
    "K","L","M","N","O",
    "P","Q","R","S","T",
    "U","V","W","X","Y"
];

// fetch progres
async function loadProgress() {

    try {
        const response = await fetch(
            `${API_BASE_URL}/users/me`,
            {
                method: "GET",
                headers: {
                    "Authorization": `Bearer ${token}`
                }
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Failed");
        }

        const progress = data.recognizedLetters;

        renderGrid(progress);

    } catch (e) {
        console.error(e);
        progressText.textContent = "Error loading progress";
    }
}

function renderGrid(progress) {

    grid.innerHTML = "";

    let learnedCount = 0;

    ALPHABET.forEach(letter => {

        const learned = progress[letter];

        if (learned) learnedCount++;

        const card = document.createElement("div");
        card.className = "letter-card";

        if (learned) {
            card.classList.add("learned");
        }

        card.innerHTML = `
            <span class="letter">${letter}</span>
            <span class="check">${learned ? "✔" : ""}</span>
        `;

        card.addEventListener("click", () => {
            window.location.href = `learn-letter.html?letter=${letter}`;
        });

        grid.appendChild(card);

    });

    progressText.textContent =
        `Progress: ${learnedCount} / ${ALPHABET.length} letters learned`;
}

loadProgress();

// back button
backBtn.addEventListener("click", () => {
    window.location.href = "profile.html";
});