import React from "react";
import Navbar from "./Components/Header";
import { Route, Routes, useLocation } from "react-router-dom";
import Home from "./Home";
import Footer from "./Components/Footer";
import Frontpage from "./Components/Frontpage";

const App = () => {
  const isOwnerPath = useLocation().pathname.includes("owner");
  return (
    <div className="app">
      {!isOwnerPath && <Navbar />}
      <div className="main-content">
        <Routes>
          <Route path="/" element={<Frontpage />} />
          <Route path="/home" element={<Home/>} />
        </Routes>
      </div>
      <Footer />
    </div>
  );
};

export default App;
