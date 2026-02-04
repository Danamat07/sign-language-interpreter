import { API_BASE_URL } from "../config/api.js";

const form = document.getElementById("forgot-form");
const message = document.getElementById("message");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const email = document.getElementById("email").value.trim();

    message.textContent = "Generating password reset...";
    message.style.color = "white";

    const url =
        `${API_BASE_URL}/users/forgot-password` +
        `?email=${encodeURIComponent(email)}`;

    try {
        const response = await fetch(url, {
            method: "POST"
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Failed to generate reset link");
        }

        message.textContent =
            "Redirecting to password reset page...";
        message.style.color = "lightgreen";

        // 🔥 DESCHIDEM LINK-UL DE RESET (RESET REAL)
        setTimeout(() => {
            window.open(data.reset_link, "_blank");
        }, 1000);

    } catch (error) {
        message.textContent = error.message;
        message.style.color = "red";
    }
});
