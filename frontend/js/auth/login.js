import { auth } from "../config/firebase.js";
import { signInWithEmailAndPassword } from
    "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js";

const form = document.getElementById("login-form");
const message = document.getElementById("message");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const email = document.getElementById("email").value.trim();
    const password = document.getElementById("password").value;

    message.textContent = "Logging in...";
    message.style.color = "white";

    try {
        const userCredential = await signInWithEmailAndPassword(
            auth,
            email,
            password
        );

        const user = userCredential.user;

        // optional: salvam token-ul pentru request-uri viitoare
        const token = await user.getIdToken();
        localStorage.setItem("firebaseToken", token);

        message.textContent = "Login successful!";
        message.style.color = "lightgreen";

        setTimeout(() => {
            window.location.href = "profile.html";
        }, 1000);

    } catch (error) {
        message.textContent = error.message;
        message.style.color = "red";
    }
});
