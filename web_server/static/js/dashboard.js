// ================================
// Update Status Panel
// ================================
async function updateStatus() {
    try {
        let res = await fetch("/status");
        let data = await res.json();

        // --- Update text fields ---
        document.getElementById("fps").innerText = data.fps.toFixed(1);
        document.getElementById("conf").innerText = data.confidence.toFixed(2);
        document.getElementById("last_seen").innerText = data.last_seen;
        document.getElementById("uptime").innerText = data.uptime;

        let tempElem = document.getElementById("temp");
        tempElem.innerText = data.cpu_temp !== "--" ? data.cpu_temp : "--";

        // --- Person indicator ---
        let indicator = document.getElementById("person-indicator");
        if (data.person_count > 0) {
            indicator.classList.add("on");
            indicator.classList.remove("off");
            indicator.innerText = `PERSON DETECTED (${data.person_count})`;
        } else {
            indicator.classList.remove("on");
            indicator.classList.add("off");
            indicator.innerText = "NO PERSON";
        }

        // --- Event Log ---
        let log = document.getElementById("event-log");
        log.innerHTML = ""; // clear existing

        data.events.forEach((evt) => {
            let item = document.createElement("li");
            item.innerText = evt;
            log.appendChild(item);
        });

    } catch (err) {
        console.error("Status update failed:", err);
    }
}

// Refresh panels
setInterval(updateStatus, 800);
