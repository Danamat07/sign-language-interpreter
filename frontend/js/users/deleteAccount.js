import { API_BASE_URL } from "../config/api.js";
import { auth } from "../config/firebase.js";
import { signOut } from
    "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js";

const deleteBtn = document.getElementById("confirm-delete");
const message = document.getElementById("message");

const token = localStorage.getItem("firebaseToken");

if (!token) {
    window.location.href = "index.html";
}

deleteBtn.addEventListener("click", async () => {
    const confirmed = confirm(
        "Are you absolutely sure you want to delete your account?"
    );

    if (!confirmed) return;

    message.textContent = "Deleting account...";
    message.style.color = "white";

    try {
        const response = await fetch(
            `${API_BASE_URL}/users/delete-account`,
            {
                method: "DELETE",
                headers: {
                    "Authorization": `Bearer ${token}`
                }
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Failed to delete account");
        }

        // logout local firebase session
        try {
            await signOut(auth);
        } catch (e) {
            // ignore
        }

        localStorage.removeItem("firebaseToken");

        message.textContent = "Account deleted successfully.";
        message.style.color = "lightgreen";

        setTimeout(() => {
            window.location.href = "index.html";
        }, 1500);

    } catch (error) {
        message.textContent = error.message;
        message.style.color = "red";
    }
});
