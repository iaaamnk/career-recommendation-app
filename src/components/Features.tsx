import React from 'react';

export function Features() {
    return (
        <section id="features" className="features-section">
            <div className="container">
                <div className="section-header text-center">
                    <h2>Why Choose CareerCompass?</h2>
                    <p>We use data-driven insights to guide your future</p>
                </div>
                <div className="features-grid">
                    <div className="feature-card">
                        <div className="feature-icon"><i className="fas fa-brain"></i></div>
                        <h3>AI Analysis</h3>
                        <p>Our model is trained on thousands of successful career profiles to find patterns that match yours.</p>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon"><i className="fas fa-road"></i></div>
                        <h3>Clear Roadmaps</h3>
                        <p>Don't just get a job title. Get a step-by-step guide on skills, courses, and certifications.</p>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon"><i className="fas fa-chart-line"></i></div>
                        <h3>Market Insights</h3>
                        <p>Stay ahead with up-to-date salary ranges and future demand predictions for every role.</p>
                    </div>
                </div>
            </div>
        </section>
    );
}
