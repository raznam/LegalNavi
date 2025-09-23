import React, { useState } from 'react';
// Assuming you have a CSS file for styling your FAQ section
  

const Faq = () => {
  // Define your FAQ data and their active state using useState
  const [faqs, setFaqs] = useState([
    {
      id: 1,
      question: "How accurate is Legal Navi's legal information?",
      answer: "Legal Navi uses verified legal databases and is regularly updated by our team of legal experts. While we strive for 100% accuracy, we recommend consulting with a qualified attorney for critical legal matters. Our AI provides general guidance but shouldn't replace professional legal advice.",
      isActive: false,
    },
    {
      id: 2,
      question: "What makes Legal Navi different from other legal research tools?",
      answer: "Unlike traditional legal research platforms, Legal Navi combines AI-powered analysis with intuitive visualization tools. Our system provides not just case law and statutes, but also summarizes complex legal concepts, checks document completeness, and helps visualize case timelines - all in one integrated platform.",
      isActive: false,
    },
    {
      id: 3,
      question: "Is my legal data secure with Legal Navi?",
      answer: "Absolutely. We use end-to-end encryption for all data transmission and storage. Your documents and research remain confidential, with enterprise-grade security measures including two-factor authentication, regular security audits, and compliance with global data protection regulations.",
      isActive: false,
    },
    {
      id: 4,
      question: "Can Legal Navi help me draft legal documents?",
      answer: "Yes! Our document drafting assistant provides templates and guided creation for common legal documents. The AI suggests relevant clauses based on your jurisdiction and case specifics, though we always recommend having any drafted documents reviewed by a licensed attorney.",
      isActive: false,
    },
    {
      id: 5,
      question: "How often is the legal database updated?",
      answer: "Our primary legal databases update in real-time as new cases and legislation are published. We conduct full system updates weekly to incorporate AI model improvements and new features. Subscribers receive notifications about significant legal changes relevant to their practice areas.",
      isActive: false,
    },
  ]);

  // Function to toggle the 'isActive' state of a specific FAQ item
  const toggleFAQ = (id) => {
    setFaqs(
      faqs.map((faq) =>
        faq.id === id ? { ...faq, isActive: !faq.isActive } : faq
      )
    );
  };

  return (
    <section className="faq" id="faq">
      <h2 className="section-title">Frequently Asked Questions</h2>
      <div className="faq-container">
        {faqs.map((faq) => (
          <div
            key={faq.id} // Important for React list rendering
            className={`faq-item ${faq.isActive ? 'active' : ''}`} // Conditionally add 'active' class
          >
            <div className="faq-question" onClick={() => toggleFAQ(faq.id)}>
              <span>{faq.question}</span>
              {/* You can still use Font Awesome. Make sure it's linked in your project. */}
              <i className="fas fa-chevron-down"></i> 
            </div>
            {/* Conditionally render the answer based on isActive state */}
            {faq.isActive && (
              <div className="faq-answer">
                <div className="faq-answer-content">
                  {faq.answer}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
};

export default Faq;