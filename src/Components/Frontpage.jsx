import { useClerk, useUser } from "@clerk/clerk-react"; // Import useUser
import React, { useEffect } from "react"; // Import useEffect
import { useNavigate } from "react-router-dom";
import NavForFrontPage from "./NavForFrontpage";


const Frontpage = () => {
  const navigate = useNavigate();
  const { openSignIn } = useClerk();
  const { isSignedIn } = useUser(); // Get isSignedIn status
  

  const goToMain = () => {
    navigate("/home");
  };

  // Use useEffect to watch for isSignedIn status
  useEffect(() => {
    if (isSignedIn) {
      goToMain(); // Navigate if the user is signed in
    }
  }, [isSignedIn, goToMain]); // Depend on isSignedIn and goToMain

  return (
    
    <div>
      <div>
         <NavForFrontPage/>
      </div>
      <div className="landing-wrapper">
        <h1>
          Introducing <span className="glow">Legal Navi</span>
        </h1>
        <div className="intro-buttons"></div>
        <div className="intro-buttons">
          <button type="button" onClick={openSignIn}>
            <i className="fas fa-rocket"></i> Login to Try Legal Navi
          </button>
          
        </div>
        
        <p className="intro-text">
          We’ve trained a model called <strong>Legal Navi</strong> that
          interacts in a conversational way. It can answer legal questions,
          guide users in filing FIRs, summarize BNS laws, and ensure legal
          document completeness. Legal Navi has been built using real-world
          legal data and deep NLP models for accurate, friendly responses.
        </p>

        <section className="article-section">
          <h2>How We Trained the Model</h2>
          <p>
            Legal Navi uses transformer-based models like BERT and
            legal-specific datasets, including FIR archives, Indian Penal Code
            data, and judgment corpora. It is fine-tuned to understand legal
            jargon, sections, and context, making it suitable for legal research
            assistance.
          </p>

          <h2>Why Legal Navi?</h2>
          <p>
            Unlike general chatbots, Legal Navi is trained exclusively on Indian
            legal data and includes a special FIR checker, BNS query resolver,
            and real-time legal guidance. This makes it ideal for both students
            and citizens who need basic legal clarity.
          </p>

          <h2>What's Next?</h2>
          <p>
            Our team is working on extending Legal Navi to support document
            upload, live voice response, and integration with court data
            portals. Your feedback is valuable — use Legal Navi and tell us what
            features matter most!
          </p>
        </section>
      </div>
    </div>
  );
};

export default Frontpage;