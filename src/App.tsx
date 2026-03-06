import { useState, useRef } from "react"
import { Navbar } from "./components/Navbar"
import { CareerHero } from "./components/CareerHero"
import { ResultsSection } from "./components/ResultsSection"
import { Features } from "./components/Features"
import "./index.css"

function App() {
    const [predictionData, setPredictionData] = useState<any>(null)
    const resultsRef = useRef<HTMLDivElement>(null)

    const handlePredict = (data: any) => {
        setPredictionData(data)
        setTimeout(() => {
            // scroll to results section
            const element = document.getElementById('result-section');
            if (element) {
                element.scrollIntoView({ behavior: 'smooth' });
            }
        }, 100)
    }

    return (
        <div className="min-h-screen">
            <Navbar />

            <main>
                <CareerHero onPredict={handlePredict} />

                {predictionData && (
                    <ResultsSection data={predictionData} />
                )}

                <Features />
            </main>

            <footer>
                <div className="container">
                    <div className="footer-content">
                        <div className="footer-brand">
                            <h4>CareerCompassAI</h4>
                            <p>Guiding your professional journey with intelligence.</p>
                        </div>
                        <div className="footer-links">
                            <a href="#">Privacy Policy</a>
                            <a href="#">Terms of Service</a>
                            <a href="#">Contact</a>
                        </div>
                    </div>
                    <div className="footer-bottom">
                        <p>&copy; 2025 CareerCompassAI. All rights reserved.</p>
                    </div>
                </div>
            </footer>
        </div>
    )
}

export default App
