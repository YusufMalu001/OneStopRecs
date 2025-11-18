import React from "react";
import "./Navbar.css";
import logo from './assets/logo.png';

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="logo-container">
        <img src={logo} alt="Logo" />
      </div>
      <ul>
        <li><a href="#">About</a></li>
        <li><a href="#">More Apps</a></li>
        <li><a href="#">Share</a></li>
        <li><a href="#">Contact</a></li>
      </ul>
    </nav>
  );
};

export default Navbar;
