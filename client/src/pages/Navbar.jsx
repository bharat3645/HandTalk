import React from "react";
import { Link } from "react-router-dom";

const Navbar = () => {
  return (
    <nav className="asl-navbar">
      <div className="asl-container">
        <Link to="/" className="asl-logo">HandTalk</Link>
        <ul className="asl-nav-links">
          <li><Link to="/">Home</Link></li>
          <li><Link to="/self-testing">Self Testing</Link></li>
          <li><Link to="/video-calling">Video Calling</Link></li>
          <li><Link to="/learn">Learn ASL</Link></li>
          <li><Link to="/explore">Explore Model</Link></li>
        </ul>
      </div>
    </nav>
  );
};

export default Navbar;
