# 🌦️ WeatherForecast

A simple and responsive web application that displays current weather information for any city using real-time data from a public weather API. Built with HTML, CSS, and JavaScript, this project demonstrates API integration, DOM manipulation, and asynchronous JavaScript.

---

## 🚀 Features

- 🌍 Search weather by city name
- ☀️ Displays temperature, weather conditions, and location
- 🕒 Real-time API data fetching
- 📱 Responsive UI suitable for mobile and desktop
- ❗ Error handling for invalid input

---

## 🛠️ Tech Stack

- HTML5
- CSS3
- JavaScript (Vanilla)
- Weather API (OpenWeatherMap or similar)

---

## 📁 Project Structure

```
Weatherforecast/
│
├── index.html        # Main UI
├── styles.css        # CSS styles
├── script.js         # JavaScript logic
└── README.md
```

---

## 🔧 Features Explained

✔ **API Integration**  
Fetches real-time weather data using a public weather API.

✔ **Search Functionality**  
Users can enter any city name to get current weather details.

✔ **Responsive Design**  
UI adapts for both desktop and mobile screens.

✔ **Graceful Error Handling**  
Shows alerts when city is not found or API request fails.

---

## 🚀 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ayushi227/Weatherforecast.git
   cd Weatherforecast
   ```

2. **Open in browser**
   - Double-click `index.html`, or  
   - Use a local server (e.g., VS Code Live Server)

```bash
# Optional: run a simple local server
npx http-server
```

---

## 🔑 Requirements

You’ll need:
- A modern web browser
- (Optional) A weather API key if using a key-protected API

If using an API like OpenWeatherMap:
1. Sign up for a free API key
2. Replace the placeholder API key in `script.js`

---

## 🧠 How It Works

1. User enters a city name and submits.
2. `script.js` sends a request to the weather API.
3. The API returns weather data in JSON.
4. Weather details are extracted and shown on the page.

Example info shown:
- City & Country
- Temperature
- Weather description

---

## 📌 Future Improvements

- Add weather icons based on conditions
- Show 5-day forecast
- Use local storage to save recent searches
- Deploy as a PWA (Progressive Web App)

---

## 👩‍💻 Author

Ayushi Khare  
GitHub: https://github.com/Ayushi227  
LinkedIn: https://www.linkedin.com/in/ayushi-khare-083b5b205/

---

⭐ **If you enjoyed this project, please give it a star!**
