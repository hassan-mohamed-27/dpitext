document.getElementById('emotionForm').addEventListener('submit', async function (event) {
    event.preventDefault();
    const userInput = document.getElementById('userInput').value;

    const response = await fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: userInput })
    });

    const result = await response.json();
    document.getElementById('result').textContent = `Emotion: ${result.emotion}`;
});
