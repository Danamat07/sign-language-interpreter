import { API_BASE_URL } from "../config/api.js";

const usernameEl = document.getElementById("username");
const emailEl = document.getElementById("email");
const message = document.getElementById("message");

const token = localStorage.getItem("firebaseToken");

if (!token) {
    window.location.href = "index.html";
}

async function loadProfile() {
    try {
        const response = await fetch(`${API_BASE_URL}/users/me`, {
            method: "GET",
            headers: {
                "Authorization": `Bearer ${token}`
            }
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Failed to load profile");
        }

        usernameEl.textContent = data.username;
        emailEl.textContent = data.email;

    } catch (error) {
        message.textContent = error.message;
        message.style.color = "red";

        // token invalid / expirat
        localStorage.removeItem("firebaseToken");
        setTimeout(() => {
            window.location.href = "index.html";
        }, 1500);
    }
}

loadProfile();
