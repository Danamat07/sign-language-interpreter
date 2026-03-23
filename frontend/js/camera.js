import { API_BASE_URL } from "./config/api.js";

// ================= VIDEO =================
const video = document.createElement("video");
video.autoplay = true;
video.playsInline = true;
video.style.display = "none";
document.body.appendChild(video);

// ================= UI =================
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");

const letterEl = document.getElementById("predicted-letter");
const confidenceEl = document.getElementById("confidence");
const messageEl = document.getElementById("message");

const targetEl = document.getElementById("target-letter");   // countdown / instructions
const gameLetterEl = document.getElementById("game-letter"); // litera jocului

const streakEl = document.getElementById("game-streak");
const timerEl = document.getElementById("game-timer");
const highscoreEl = document.getElementById("game-highscore");

const instructionsBox = document.getElementById("instructions-box");

const startBtn = document.getElementById("start-game-btn");
const backBtn = document.getElementById("back-btn");
const speechBtn = document.getElementById("speech-btn");

const token = localStorage.getItem("firebaseToken");

if (!token) window.location.href = "index.html";

// ================= CANVAS =================
overlay.width = 640;
overlay.height = 480;

// ================= BUFFERS =================
const VOTING_FRAMES = 7;
const predictionBuffer = [];
const confidenceBuffer = [];

// ================= GAME =================
const ALPHABET = [
    "A","B","C","D","E",
    "F","G","H","I",
    "K","L","M","N","O",
    "P","Q","R","S","T",
    "U","V","W","X","Y"
];

let gameActive = false;
let currentTarget = null;
let previousTarget = null;

let streak = 0;
let timeLeft = 60;
let lastCorrect = null;

let highscore = 0;

// ================= SPEECH =================
let speechEnabled = false;
let lastSpokenLetter = null;

speechBtn.addEventListener("click", () => {
    speechEnabled = !speechEnabled;
    speechBtn.textContent = speechEnabled ? "Speech: ON" : "Speech: OFF";
});

function speakLetter(letter) {
    if (!speechEnabled) return;
    if (!letter || letter === "-") return;
    if (letter === lastSpokenLetter) return;

    lastSpokenLetter = letter;

    const utterance = new SpeechSynthesisUtterance(letter);
    utterance.rate = 0.6;
    utterance.lang = "en-US";

    speechSynthesis.cancel();
    speechSynthesis.speak(utterance);
}

// ================= INSTRUCTIONS =================
instructionsBox.textContent = `
• A random letter will appear.
• Mimic the sign with your hand.
• Keep it steady until recognized.
• Build your streak in 60 seconds.
• Try to beat your highscore!

Press "Start Game" to begin.
`;

// ================= LOAD HIGHSCORE =================
async function loadHighscore() {
    try {
        const res = await fetch(`${API_BASE_URL}/users/me`, {
            headers: {
                "Authorization": `Bearer ${token}`
            }
        });

        const data = await res.json();

        highscore = data.highscore || 0;
        highscoreEl.textContent = `Highscore: ${highscore}`;

    } catch (e) {
        console.error("Highscore load failed");
    }
}

// ================= START GAME =================
startBtn.addEventListener("click", async () => {

    await loadHighscore();

    streak = 0;
    timeLeft = 60;
    gameActive = false;
    lastCorrect = null;
    previousTarget = null;

    streakEl.textContent = "Streak: 0";
    timerEl.textContent = "Time: 60s";
    messageEl.textContent = "";

    instructionsBox.textContent = "";

    let count = 3;

    // countdown
    targetEl.textContent = `Get Ready: ${count}`;

    const interval = setInterval(() => {

        count--;

        if (count > 0) {
            targetEl.textContent = `Get Ready: ${count}`;
        } else {
            clearInterval(interval);
            startGame();
            targetEl.textContent = " ";
        }

    }, 1000);
});

// ================= GAME START =================
function startGame() {
    gameActive = true;

    pickRandomLetter();

    // litera apare DOAR aici
    gameLetterEl.textContent = currentTarget;

    startTimer();
}

// ================= RANDOM LETTER =================
function pickRandomLetter() {
    let newLetter;

    do {
        const i = Math.floor(Math.random() * ALPHABET.length);
        newLetter = ALPHABET[i];
    } while (newLetter === previousTarget);

    previousTarget = newLetter;
    currentTarget = newLetter;

    gameLetterEl.textContent = currentTarget;
}

// ================= TIMER =================
function startTimer() {
    const interval = setInterval(() => {

        if (!gameActive) {
            clearInterval(interval);
            return;
        }

        timeLeft--;
        timerEl.textContent = `Time: ${timeLeft}s`;

        if (timeLeft <= 0) {
            clearInterval(interval);
            endGame();
        }

    }, 1000);
}

// ================= END GAME =================
async function endGame() {

    gameActive = false;

    targetEl.textContent = "Game Over";
    gameLetterEl.textContent = "-";

    messageEl.textContent = `Score: ${streak}`;

    try {
        await fetch(`${API_BASE_URL}/users/save-score`, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${token}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ score: streak })
        });

        await loadHighscore();

    } catch (e) {
        console.error("Save failed");
    }
}

// ================= CAMERA =================
async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true
    });

    video.srcObject = stream;
}

startCamera();

// ================= LOOP =================
async function predictFrame() {

    if (video.readyState < 2) {
        requestAnimationFrame(predictFrame);
        return;
    }

    const tmpCanvas = document.createElement("canvas");
    tmpCanvas.width = video.videoWidth;
    tmpCanvas.height = video.videoHeight;

    const tmpCtx = tmpCanvas.getContext("2d");

    tmpCtx.translate(tmpCanvas.width, 0);
    tmpCtx.scale(-1, 1);
    tmpCtx.drawImage(video, 0, 0);

    const blob = await new Promise(res =>
        tmpCanvas.toBlob(res, "image/jpeg")
    );

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

        let letter = data.letter || "";
        let confidence = data.confidence || 0;

        predictionBuffer.push(letter);
        confidenceBuffer.push(confidence);

        if (predictionBuffer.length > VOTING_FRAMES) {
            predictionBuffer.shift();
            confidenceBuffer.shift();
        }

        let stableLetter = "-";
        let avgConfidence = 0;

        if (predictionBuffer.length > 0) {
            const counts = {};
            predictionBuffer.forEach(l => counts[l] = (counts[l] || 0) + 1);

            stableLetter = Object.keys(counts)
                .reduce((a, b) => counts[a] > counts[b] ? a : b);

            avgConfidence =
                confidenceBuffer.reduce((a, b) => a + b, 0) / confidenceBuffer.length;
        }

        letterEl.textContent = `Letter: ${stableLetter}`;
        confidenceEl.textContent = `Confidence: ${(avgConfidence * 100).toFixed(2)}%`;

        speakLetter(stableLetter);

        // GAME LOGIC
        if (
            gameActive &&
            stableLetter === currentTarget &&
            avgConfidence > 0.8 &&
            lastCorrect !== currentTarget
        ) {
            streak++;
            streakEl.textContent = `Streak: ${streak}`;

            if (streak > highscore) {
                highscore = streak;
                highscoreEl.textContent = `Highscore: ${highscore}`;
            }

            lastCorrect = currentTarget;
            pickRandomLetter();
        }

        // DRAW
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
        messageEl.textContent = "Prediction failed";
    }

    requestAnimationFrame(predictFrame);
}

video.addEventListener("loadeddata", predictFrame);

// ================= BACK =================
backBtn.addEventListener("click", () => {
    window.location.href = "profile.html";
});