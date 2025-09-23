 import { useClerk, useUser,UserButton } from '@clerk/clerk-react'
import React from 'react'
import { use } from 'react'
 

 const NavForFrontPage = () => {

    const {openSignIn} = useClerk()
     const {user} = useUser() 
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
      
   return (
    <header className="nav">
    <div className="logo">
      <i className="fas fa-balance-scale"></i>
      <span>Legal Navi</span>
    </div>
    
   
    <div className='hidden md:flex items-centergap-4'>
      {user ? (<UserButton></UserButton>):(<button onClick={openSignIn} className="cursor-pointer px-8 py-2  transition text-white rounded-full">
                    Login
      </button>)}
      
    </div>
  </header>
   )
 }
 
 export default NavForFrontPage 