import React, { useState } from 'react';
import { BeamsBackground } from "@/components/ui/beams-background";

interface FormData {
    age: string;
    education: string;
    skills: string;
    interests: string;
    riasec: number[];
}

export function CareerHero({ onPredict }: { onPredict: (data: any) => void }) {
    const [formData, setFormData] = useState<FormData>({
        age: '',
        education: '',
        skills: '',
        interests: '',
        riasec: [5, 5, 5, 5, 5, 5],
    });
    const [loading, setLoading] = useState(false);

    const riasecLabels = [
        { label: 'Realistic', name: 'r-score', desc: 'Do you like playing sports, tinkering with gadgets, taking care of plants or pets, or having outdoor adventures?' },
        { label: 'Investigative', name: 'i-score', desc: 'Do you enjoy learning cool stuff, solving mysteries, or figuring things out with friends?' },
        { label: 'Artistic', name: 'a-score', desc: 'Do you love making art, coming up with awesome ideas, or just letting your imagination run wild?' },
        { label: 'Social', name: 's-score', desc: 'Do you enjoy sharing cool facts and helping and teaching others?' },
        { label: 'Enterprising', name: 'e-score', desc: 'Do you like to be in charge, take the lead, and make big things happen?' },
        { label: 'Conventional', name: 'c-score', desc: 'Do you like working with numbers, organizing stuff, and following steps to get things done?' },
    ];

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);

        const payload = {
            age: parseInt(formData.age),
            education: formData.education,
            skills: formData.skills.split(',').map(s => s.trim()).filter(s => s),
            interests: formData.interests.split(',').map(s => s.trim()).filter(s => s),
            riasec_scores: formData.riasec,
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (response.ok) {
                onPredict(data);
            } else {
                alert(data.error || 'Prediction failed');
            }
        } catch (error) {
            console.error(error);
            alert('Error connecting to backend');
        } finally {
            setLoading(false);
        }
    };

    return (
        <header className="hero">
            <BeamsBackground className="absolute inset-0 z-0" />

            <div className="hero-container">
                <div className="hero-text">
                    <div className="badge">AI-Powered Guidance</div>
                    <h1>Discover Your Perfect Career Path</h1>
                    <p className="hero-subtitle">Unsure about your future? Our advanced AI analyzes your skills, interests, and personality to recommend the best career opportunities for you.</p>

                    <div className="trust-badges">
                        <div className="trust-item"><i className="fas fa-check-circle"></i> 98% Accuracy</div>
                        <div className="trust-item"><i className="fas fa-user-shield"></i> 100% Private</div>
                    </div>

                    <div className="hero-features">
                        <div className="hf-item">
                            <div className="hf-icon"><i className="fas fa-brain"></i></div>
                            <div className="hf-text">
                                <strong>AI Personality Analysis</strong>
                                <p>Deep psychological profiling with RIASEC model</p>
                            </div>
                        </div>
                        <div className="hf-item">
                            <div className="hf-icon"><i className="fas fa-map-signs"></i></div>
                            <div className="hf-text">
                                <strong>Step-by-Step Roadmaps</strong>
                                <p>Courses, skills, and certifications you need</p>
                            </div>
                        </div>
                        <div className="hf-item">
                            <div className="hf-icon"><i className="fas fa-chart-line"></i></div>
                            <div className="hf-text">
                                <strong>Market Data & Trends</strong>
                                <p>Real salary ranges and future demand scores</p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Input Form Card */}
                <div className="hero-card" id="input-card">
                    <div className="card-header">
                        <h2>Career Discovery Engine</h2>
                        <p>Complete your profile to get started</p>
                    </div>

                    <form onSubmit={handleSubmit} id="career-form">
                        <div className="form-grid">
                            <div className="form-group">
                                <label htmlFor="age">Age</label>
                                <input
                                    type="number" id="age" name="age" min="16" max="80"
                                    required placeholder="24"
                                    value={formData.age}
                                    onChange={e => setFormData({ ...formData, age: e.target.value })}
                                />
                            </div>

                            <div className="form-group">
                                <label htmlFor="education">Education</label>
                                <select
                                    id="education" name="education" required
                                    value={formData.education}
                                    onChange={e => setFormData({ ...formData, education: e.target.value })}
                                >
                                    <option value="" disabled selected>Select Level</option>
                                    <option value="High School">High School</option>
                                    <option value="Bachelor's">Bachelor's Degree</option>
                                    <option value="Master's">Master's Degree</option>
                                    <option value="PhD">PhD</option>
                                    <option value="Diploma">Diploma</option>
                                </select>
                            </div>
                        </div>

                        <div className="form-group">
                            <label htmlFor="skills">Key Skills</label>
                            <input
                                type="text" id="skills" name="skills" required placeholder="e.g., Python, SQL, Design"
                                value={formData.skills}
                                onChange={e => setFormData({ ...formData, skills: e.target.value })}
                            />
                        </div>

                        <div className="form-group">
                            <label htmlFor="interests">Interests</label>
                            <input
                                type="text" id="interests" name="interests" required placeholder="e.g., Tech, Art, Finance"
                                value={formData.interests}
                                onChange={e => setFormData({ ...formData, interests: e.target.value })}
                            />
                        </div>

                        <div className="form-group">
                            <label className="mb-2">Personality Profile (RIASEC)</label>
                            <div className="riasec-grid">
                                {riasecLabels.map((item, idx) => (
                                    <div key={idx} className="range-wrapper">
                                        <div className="range-label">
                                            <span>{item.label}</span>
                                            <span className="val-badge">{formData.riasec[idx]}</span>
                                        </div>
                                        <p className="riasec-desc">{item.desc}</p>
                                        <input
                                            type="range" min="1" max="10"
                                            value={formData.riasec[idx]}
                                            onChange={e => {
                                                const newRiasec = [...formData.riasec];
                                                newRiasec[idx] = parseInt(e.target.value);
                                                setFormData({ ...formData, riasec: newRiasec });
                                            }}
                                        />
                                    </div>
                                ))}
                            </div>
                        </div>

                        <button type="submit" className="btn-submit" disabled={loading}>
                            {loading ? "Analyzing..." : <>Analyze Profile <i className="fas fa-arrow-right"></i></>}
                        </button>
                    </form>
                </div>
            </div>
        </header>
    );
}
