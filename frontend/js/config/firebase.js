import { initializeApp } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js";
import { getAuth } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js";

const firebaseConfig = {
  apiKey: "AIzaSyCsA02OoYmSh4mSNGmlW1RS1k00y6b761U",
  authDomain: "asl-recognition-8a208.firebaseapp.com",
  projectId: "asl-recognition-8a208",
  storageBucket: "asl-recognition-8a208.firebasestorage.app",
  messagingSenderId: "39767537082",
  appId: "1:39767537082:web:d8617e640c7a6972301983"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
