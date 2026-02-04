import { API_BASE_URL } from "../config/api.js";

const form = document.getElementById("signup-form");
const message = document.getElementById("message");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const username = document.getElementById("username").value.trim();
    const email = document.getElementById("email").value.trim();
    const password = document.getElementById("password").value;

    message.textContent = "Creating account...";
    message.style.color = "white";

    const url =
        `${API_BASE_URL}/users/signup` +
        `?email=${encodeURIComponent(email)}` +
        `&password=${encodeURIComponent(password)}` +
        `&username=${encodeURIComponent(username)}`;

    try {
        const response = await fetch(url, {
            method: "POST"
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Signup failed");
        }

        message.textContent = "Account created successfully!";
        message.style.color = "lightgreen";

        setTimeout(() => {
            window.location.href = "index.html";
        }, 1500);

    } catch (error) {
        message.textContent = error.message;
        message.style.color = "red";
    }
});
