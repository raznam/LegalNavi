console.log("script.js loaded"); // Diagnostic: Confirm script is running

// Navbar scroll effect
const nav = document.querySelector('.nav');
if (nav) {
  window.addEventListener('scroll', () => {
    console.log("Scroll event triggered"); // Diagnostic
    if (window.scrollY > 50) {
      nav.classList.add('scrolled');
    } else {
      nav.classList.remove('scrolled');
    }
  });
} else {
  console.error("Error: .nav element not found");
}

// Voice recognition functionality
const micBtn = document.getElementById('micBtn');
const input = document.getElementById('searchInput');
const ctaButton = document.querySelector('.cta');

if (!micBtn || !input || !ctaButton) {
  console.error("Error: One or more elements not found", { micBtn, input, ctaButton });
}

let recognition = null;
if ('webkitSpeechRecognition' in window) {
  console.log("webkitSpeechRecognition is supported");
  recognition = new webkitSpeechRecognition();
  recognition.lang = 'en-US';
  recognition.continuous = false;
  recognition.interimResults = false;

  micBtn.addEventListener('click', () => {
    console.log("Mic button clicked"); // Diagnostic
    if (micBtn.classList.contains('listening')) {
      recognition.stop();
      micBtn.classList.remove('listening');
      input.placeholder = "Ask about legal topics, FIR checks, or case summaries...";
      return;
    }

    micBtn.classList.add('listening');
    input.placeholder = "Listening... Speak now";
    recognition.start();
  });

  recognition.onresult = function(event) {
    const transcript = event.results[0][0].transcript;
    console.log("Voice recognition result:", transcript); // Diagnostic
    input.value = transcript;
    micBtn.classList.remove('listening');
    input.placeholder = "Ask about legal topics, FIR checks, or case summaries...";
  };

  recognition.onerror = function(event) {
    console.error("Voice recognition error:", event.error); // Improved error logging
    micBtn.classList.remove('listening');
    input.placeholder = "Ask about legal topics, FIR checks, or case summaries...";
  };

  recognition.onend = function() {
    console.log("Voice recognition ended"); // Diagnostic
    if (micBtn.classList.contains('listening')) {
      micBtn.classList.remove('listening');
      input.placeholder = "Ask about legal topics, FIR checks, or case summaries...";
    }
  };
} else {
  console.warn("webkitSpeechRecognition not supported");
  micBtn.addEventListener('click', () => {
    alert('Voice input is not supported in your browser. Please try Chrome or Edge.');
  });
}

// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function(e) {
    console.log("Anchor clicked:", this.getAttribute('href')); // Diagnostic
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      target.scrollIntoView({
        behavior: 'smooth'
      });
    } else {
      console.error("Target not found:", this.getAttribute('href'));
    }
  });
});

// CTA button animation
if (ctaButton) {
  ctaButton.addEventListener('mouseenter', () => {
    console.log("CTA mouseenter"); // Diagnostic
    ctaButton.querySelector('i').style.transform = 'rotate(-30deg) scale(1.2)';
  });
  ctaButton.addEventListener('mouseleave', () => {
    console.log("CTA mouseleave"); // Diagnostic
    ctaButton.querySelector('i').style.transform = 'rotate(0deg) scale(1)';
  });
}

// Input focus enhancement
if (input) {
  input.addEventListener('focus', () => {
    console.log("Input focused"); // Diagnostic
    input.parentElement.style.transform = 'translateY(-2px)';
  });
  input.addEventListener('blur', () => {
    console.log("Input blurred"); // Diagnostic
    input.parentElement.style.transform = 'translateY(0)';
  });
}

// Floating animation for features
const features = document.querySelectorAll('.feature');
features.forEach((feature, index) => {
  feature.style.transitionDelay = `${index * 0.1}s`;
  console.log(`Feature ${index} transition delay set`); // Diagnostic
});

// Glow effect for headings
const headings = document.querySelectorAll('h1, h2, h3');
headings.forEach(heading => {
  heading.addEventListener('mouseover', () => {
    console.log("Heading mouseover:", heading.textContent); // Diagnostic
    heading.classList.add('glow');
  });
  heading.addEventListener('mouseout', () => {
    console.log("Heading mouseout:", heading.textContent); // Diagnostic
    heading.classList.remove('glow');
  });
});

// FAQ toggle functionality
const faqQuestions = document.querySelectorAll('.faq-question');
if (faqQuestions.length > 0) {
  faqQuestions.forEach(item => {
    item.addEventListener('click', () => {
      console.log("FAQ question clicked:", item.textContent); // Diagnostic
      const faqItem = item.parentElement;
      faqItem.classList.toggle('active');
    });
  });
} else {
  console.error("Error: No .faq-question elements found");
}