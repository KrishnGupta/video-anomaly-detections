// Function to add flip animation every 2-3 seconds
function startFlipping() {
    const overlay = document.querySelector('.alert-overlay');
    setInterval(() => {
        overlay.classList.toggle('');
    }, 2000);
}

// Function to handle OK button click, hide the alert, and stop the alert sound
function handleOkButtonClick() {
    const overlay = document.querySelector('.alert-overlay');
    const alertSound = document.querySelector('#alert-sound');
    
    if (overlay) {
        overlay.style.display = 'none'; // Hide the overlay
    }
    
    if (alertSound) {
        alertSound.pause(); // Stop playing the sound
        alertSound.currentTime = 0; // Reset the sound to the beginning
    }
}

// Play the alert sound and start flipping animation if confidence is low
document.addEventListener('DOMContentLoaded', () => {
    if (document.querySelector('.alert-overlay')) {
        startFlipping();
        document.querySelector('#ok-button').addEventListener('click', handleOkButtonClick);
    }
});
