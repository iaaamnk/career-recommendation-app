import React from 'react';

export function Navbar() {
    return (
        <nav className="navbar">
            <div className="nav-container">
                <div className="logo">
                    <div className="logo-icon"><i className="fas fa-compass"></i></div>
                    <span>CareerCompass<span className="text-primary">AI</span></span>
                </div>
                <div className="nav-links">
                    <a href="/" className="active">Home</a>
                    <a href="#features">Features</a>
                    <a href="#how-it-works">How it Works</a>
                </div>
            </div>
        </nav>
    );
}
