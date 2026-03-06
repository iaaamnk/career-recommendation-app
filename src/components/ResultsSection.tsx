import React from 'react';

interface PredictionResults {
    Recommended_Career: string;
    Recommendation_Score: number;
    Explanation: string;
    Career_Roadmap: {
        "Required skills": string | string[];
        "Courses (Free)": string[];
        "Courses (Paid)": string[];
        "Certifications": string | string[];
        "Salary range in India": string;
    };
    Top_3_Careers: Array<{ career: string; score: number }>;
}

export function ResultsSection({ data }: { data: PredictionResults }) {
    if (!data) return null;

    return (
        <section id="result-section" className="results-section visible">
            <div className="container">
                <div className="section-header text-center">
                    <h2>Your Career Analysis</h2>
                    <p>Based on your unique profile and preferences</p>
                </div>

                <div className="results-card">
                    <div className="results-grid">
                        {/* Main Recommendation */}
                        <div className="main-rec">
                            <div className="rec-header">
                                <span className="match-badge">{(data.Recommendation_Score * 100).toFixed(0)}% AccurateMatch</span>
                                <h3>{data.Recommended_Career}</h3>
                            </div>
                            <div className="rec-body">
                                <h4>Why this career?</h4>
                                <p>{data.Explanation.replace(/\*\*(.*?)\*\*/g, '$1')}</p>

                                <div className="roadmap-box">
                                    <h4><i className="fas fa-map-signs"></i> Your Roadmap</h4>
                                    <div className="roadmap-item">
                                        <span className="rm-label">Skills</span>
                                        <span className="rm-value">
                                            {Array.isArray(data.Career_Roadmap["Required skills"])
                                                ? data.Career_Roadmap["Required skills"].join(", ")
                                                : data.Career_Roadmap["Required skills"]}
                                        </span>
                                    </div>
                                    <div className="roadmap-item">
                                        <span className="rm-label">Courses</span>
                                        <span className="rm-value">
                                            {data.Career_Roadmap["Courses (Free)"].slice(0, 2).join(", ")}
                                        </span>
                                    </div>
                                    <div className="roadmap-item">
                                        <span className="rm-label">Certs</span>
                                        <span className="rm-value">
                                            {Array.isArray(data.Career_Roadmap["Certifications"])
                                                ? data.Career_Roadmap["Certifications"].slice(0, 2).join(", ")
                                                : data.Career_Roadmap["Certifications"]}
                                        </span>
                                    </div>
                                    <div className="roadmap-item">
                                        <span className="rm-label">Salary (IN)</span>
                                        <span className="rm-value">{data.Career_Roadmap["Salary range in India"]}</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Alternatives */}
                        <div className="alternatives-box">
                            <h4>Alternative Paths</h4>
                            <div className="alternatives-list">
                                {data.Top_3_Careers.map((alt, i) => (
                                    <div key={i} className="alt-chip">
                                        <span>{alt.career}</span>
                                        <span className="val-badge">{(alt.score * 100).toFixed(0)}%</span>
                                    </div>
                                ))}
                            </div>
                            <div className="alt-info">
                                <p><i className="fas fa-info-circle"></i> These careers also align well with your profile but may require slightly different focus areas.</p>
                            </div>
                            <button onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })} className="btn btn-outline btn-sm mt-4">
                                Analyze Another Profile
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}
