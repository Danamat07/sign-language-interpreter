import { API_BASE_URL } from "./config/api.js";

const ctx = document.getElementById("statsChart").getContext("2d");
const messageEl = document.getElementById("message");
const token = localStorage.getItem("firebaseToken");

if (!token) {
    window.location.href = "index.html";
}

// Literele din alfabet fara J si Z
const ALPHABET = [
    "A","B","C","D","E",
    "F","G","H","I",
    "K","L","M","N","O",
    "P","Q","R","S","T",
    "U","V","W","X","Y"
];

// Configurare inițială Chart.js
const chartData = {
    labels: ALPHABET,
    datasets: [{
        label: 'Percentage of users who checked the letter',
        data: Array(ALPHABET.length).fill(0),
        backgroundColor: 'rgba(54, 162, 235, 0.7)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1
    }]
};

const statsChart = new Chart(ctx, {
    type: 'bar',
    data: chartData,
    options: {
        scales: {
            y: {
                beginAtZero: true,
                max: 100,
                ticks: {
                    callback: value => value + '%'
                }
            }
        },
        plugins: {
            legend: {
                display: true,
                onClick: null  // dezactivează toggle-ul la click pe legendă
            },
            tooltip: {
                callbacks: {
                    label: context => `${context.parsed.y.toFixed(1)}%`
                }
            }
        }
    }
});

// Funcție pentru încărcarea statisticilor
async function loadStatistics() {
    try {
        const response = await fetch(`${API_BASE_URL}/users/statistics`, {
            method: "GET",
            headers: {
                "Authorization": `Bearer ${token}`
            }
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Failed to load statistics");

        const globalStats = data.globalStats || {};
        const totalUsers = data.totalUsers || 1;

        // Calcul procent per literă
        chartData.datasets[0].data = ALPHABET.map(letter => {
            const count = globalStats[letter] || 0;
            return (count / totalUsers) * 100;
        });

        statsChart.update();

    } catch (err) {
        console.error(err);
        messageEl.textContent = err.message;
    }
}

// Load initial + refresh la fiecare 2 secunde
loadStatistics();
setInterval(loadStatistics, 2000);