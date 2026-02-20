import { API_BASE_URL } from "./config/api.js";

const video = document.getElementById("webcam");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");

const letterEl = document.getElementById("predicted-letter");
const confidenceEl = document.getElementById("confidence");

const messageEl = document.getElementById("message");
const backBtn = document.getElementById("back-btn");

const token = localStorage.getItem("firebaseToken");

if (!token) {

    window.location.href = "index.html";

}


// majority voting

const VOTING_FRAMES = 7;

const predictionBuffer = [];

const confidenceBuffer = [];


// canvas size

overlay.width = 640;

overlay.height = 480;


// start webcam

async function startCamera() {

    try {

        const stream = await navigator.mediaDevices.getUserMedia({

            video: true

        });

        video.srcObject = stream;

        video.style.transform = "scaleX(-1)";

    }

    catch (e) {

        messageEl.textContent = "Cannot access camera";

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


    const tmpCanvas = document.createElement("canvas");

    tmpCanvas.width = video.videoWidth;

    tmpCanvas.height = video.videoHeight;

    const tmpCtx = tmpCanvas.getContext("2d");


    // mirror frame

    tmpCtx.translate(tmpCanvas.width, 0);

    tmpCtx.scale(-1, 1);

    tmpCtx.drawImage(video, 0, 0);


    const blob = await new Promise(

        resolve => tmpCanvas.toBlob(resolve, "image/jpeg")

    );


    const formData = new FormData();

    formData.append("file", blob, "frame.jpg");


    try {

        const response = await fetch(

            `${API_BASE_URL}/predict`,

            {

                method: "POST",

                headers: {

                    "Authorization": `Bearer ${token}`

                },

                body: formData

            }

        );


        const data = await response.json();


        let letter = data.letter || "";

        let confidence = data.confidence || 0;


        // add to buffer

        if (letter) {

            predictionBuffer.push(letter);

            confidenceBuffer.push(confidence);

        }


        if (predictionBuffer.length > VOTING_FRAMES) {

            predictionBuffer.shift();

            confidenceBuffer.shift();

        }


        // majority voting

        let stableLetter = "-";

        let stableConfidence = 0;


        if (predictionBuffer.length > 0) {

            const counts = {};


            predictionBuffer.forEach(

                l => counts[l] = (counts[l] || 0) + 1

            );


            stableLetter = Object.keys(counts)

                .reduce(

                    (a, b) => counts[a] > counts[b] ? a : b

                );


            stableConfidence =

                confidenceBuffer.reduce(

                    (a, b) => a + b,

                    0

                ) / confidenceBuffer.length;

        }


        // update UI

        letterEl.textContent = `Letter: ${stableLetter}`;


        confidenceEl.textContent =

            `Confidence: ${(stableConfidence * 100).toFixed(2)} %`;


        // draw annotated frame

        const img = new Image();

        img.src = `data:image/jpeg;base64,${data.image}`;


        img.onload = () => {

            ctx.clearRect(

                0,

                0,

                overlay.width,

                overlay.height

            );


            ctx.save();


            ctx.scale(-1, 1);


            ctx.drawImage(

                img,

                -overlay.width,

                0,

                overlay.width,

                overlay.height

            );


            ctx.restore();

        };

    }

    catch (e) {

        console.error(e);

        messageEl.textContent = "Prediction failed";

    }


    requestAnimationFrame(predictFrame);

}


video.addEventListener(

    "loadeddata",

    predictFrame

);


backBtn.addEventListener(

    "click",

    () => {

        window.location.href = "profile.html";

    }

);