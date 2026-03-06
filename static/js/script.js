document.getElementById('career-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    // Show loader, hide form button (optional UX improvement)
    const loader = document.getElementById('loader');
    const submitBtn = this.querySelector('button[type="submit"]');

    loader.style.display = 'block';
    submitBtn.style.display = 'none';

    // Collect Data
    const age = parseInt(document.getElementById('age').value);
    const education = document.getElementById('education').value;

    // Split by comma and trim whitespace
    const skills = document.getElementById('skills').value.split(',').map(s => s.trim()).filter(s => s);
    const interests = document.getElementById('interests').value.split(',').map(s => s.trim()).filter(s => s);

    const riasec = [
        parseInt(document.getElementById('r-score').value),
        parseInt(document.getElementById('i-score').value),
        parseInt(document.getElementById('a-score').value),
        parseInt(document.getElementById('s-score').value),
        parseInt(document.getElementById('e-score').value),
        parseInt(document.getElementById('c-score').value)
    ];

    const payload = {
        age: age,
        education: education,
        skills: skills,
        interests: interests,
        riasec_scores: riasec
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Network response was not ok');
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        console.error('Error:', error);
        alert(`Error: ${error.message}`);
        submitBtn.style.display = 'block';
    } finally {
        loader.style.display = 'none';
    }
});

function displayResults(data) {
    // Hide input card on mobile, or scroll to result on desktop? 
    // Let's just show the result section.

    const resultSection = document.getElementById('result-section');

    // Populate Data
    document.getElementById('rec-career').textContent = data.Recommended_Career;
    document.getElementById('rec-score').textContent = `${(data.Recommendation_Score * 100).toFixed(1)}% Match`;
    // Parse bold markdown
    const explanationText = data.Explanation || "";
    const formattedExplanation = explanationText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    document.getElementById('rec-explanation').innerHTML = formattedExplanation;

    // Roadmap
    const roadmap = data.Career_Roadmap;
    document.getElementById('rm-skills').textContent = Array.isArray(roadmap['Required skills']) ? roadmap['Required skills'].join(', ') : roadmap['Required skills'];

    // Handle courses (could be array or string)
    const courses = [...(roadmap['Courses (Free)'] || []), ...(roadmap['Courses (Paid)'] || [])];
    document.getElementById('rm-courses').textContent = courses.length > 0 ? courses.slice(0, 3).join(', ') + '...' : 'Check online resources';

    document.getElementById('rm-certs').textContent = Array.isArray(roadmap['Certifications']) ? roadmap['Certifications'].join(', ') : roadmap['Certifications'];
    document.getElementById('rm-salary').textContent = roadmap['Salary range in India'];

    // Alternatives
    const altList = document.getElementById('alt-list');
    altList.innerHTML = '';
    data.Top_3_Careers.forEach(alt => {
        const chip = document.createElement('div');
        chip.className = 'alt-chip';
        chip.textContent = `${alt.career} (${(alt.score * 100).toFixed(0)}%)`;
        altList.appendChild(chip);
    });

    // Show Section
    resultSection.style.display = 'block';
    // Force reflow
    void resultSection.offsetWidth;
    resultSection.classList.add('visible');

    // Scroll to results
    setTimeout(() => {
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

// Handle Reset / Analyze Another Profile
document.getElementById('reset-btn').addEventListener('click', function () {
    // Reset form
    document.getElementById('career-form').reset();

    // Reset range values text
    ['r', 'i', 'a', 's', 'e', 'c'].forEach(char => {
        document.getElementById(`val-${char}`).innerText = '5';
    });

    // Hide results
    const resultSection = document.getElementById('result-section');
    resultSection.style.display = 'none';
    resultSection.classList.remove('visible');

    // Show submit button again
    const submitBtn = document.querySelector('button[type="submit"]');
    submitBtn.style.display = 'flex'; // Restore flex display

    // Scroll to form
    document.getElementById('input-card').scrollIntoView({ behavior: 'smooth' });
});
