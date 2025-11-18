import React, { useState, useEffect } from "react";
import "./TagSelector.css";  // Import your custom CSS for buttons

const TagSelector = ({ categoryType, onTagsChange }) => {
  const movieCategories = [
    "Action", "Adventure", "Comedy", "Drama", "Thriller", "Horror", "Romance",
    "Science Fiction", "Fantasy", "Mystery", "Crime", "Animation", "Documentary",
    "Musical", "Biography", "War", "Western", "Historical", "Sports", "Family",
    "Superhero", "Noir", "Short Film", "Survival", "Experimental"
  ];

  const bookCategories = [
    "Fiction", "Non-Fiction", "Mystery", "Fantasy", "Science Fiction", "Romance",
    "Historical", "Biography", "Self-Help", "Poetry"
  ];

  const productCategories = [
    "Smartphones", "Laptops", "Tablets", "Headphones", "Smartwatches",
    "Cameras", "Gaming Consoles", "TVs", "Speakers", "Wearables"
  ];

  const [selectedTags, setSelectedTags] = useState([]);

  let categories = [];
  if (categoryType === "movies") categories = movieCategories;
  else if (categoryType === "books") categories = bookCategories;
  else if (categoryType === "products") categories = productCategories;

  const toggleTag = (tag) => {
    setSelectedTags(prev => {
      if (prev.includes(tag)) {
        return prev.filter(t => t !== tag);
      } else {
        return [...prev, tag];
      }
    });
  };

  useEffect(() => {
    onTagsChange(selectedTags);
  }, [selectedTags, onTagsChange]);

  return (
    <div>
      <h3>Select {categoryType} categories</h3>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "12px" }}>
        {categories.map(tag => (
          <button
            key={tag}
            type="button"
            className={`botton ${selectedTags.includes(tag) ? "active" : ""}`}
            onClick={() => toggleTag(tag)}
          >
            {tag}
          </button>
        ))}
      </div>
    </div>
  );
};

export default TagSelector;
