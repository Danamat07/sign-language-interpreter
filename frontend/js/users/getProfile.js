import { API_BASE_URL } from "../config/api.js";


const usernameEl = document.getElementById("username");

const emailEl = document.getElementById("email");

const lettersList = document.getElementById("letters-list");

const message = document.getElementById("message");


const token = localStorage.getItem("firebaseToken");


if (!token) {

    window.location.href = "index.html";

}



const ALPHABET = [

"A","B","C","D","E",

"F","G","H","I",

"K","L","M","N","O",

"P","Q","R","S","T",

"U","V","W","X","Y"

];



function renderLetters(recognizedLetters) {

    lettersList.innerHTML = "";


    ALPHABET.forEach(letter => {


        const checked = recognizedLetters[letter];


        const item = document.createElement("div");

        item.className = "letter-item";


        item.innerHTML = `

            <label>

                <input type="checkbox"

                       disabled

                       ${checked ? "checked" : ""}>

                ${letter}

            </label>

        `;


        lettersList.appendChild(item);

    });

}



async function loadProfile() {


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

            throw new Error(

                data.detail ||

                "Failed to load profile"

            );

        }


        usernameEl.textContent = data.username;

        emailEl.textContent = data.email;


        renderLetters(

            data.recognizedLetters

        );


    }


    catch (error) {


        message.textContent = error.message;

        message.style.color = "red";


        localStorage.removeItem(

            "firebaseToken"

        );


        setTimeout(

            () => {

                window.location.href =

                "index.html";

            },

            1500

        );


    }

}



loadProfile();