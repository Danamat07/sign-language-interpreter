import { auth } from "../config/firebase.js";
import { signOut } from
    "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js";

const logoutBtn = document.getElementById("logout-btn");

logoutBtn.addEventListener("click", async () => {
    try {
        await signOut(auth);
    } catch (e) {
        // ignore
    }

    localStorage.removeItem("firebaseToken");
    window.location.href = "index.html";
});
