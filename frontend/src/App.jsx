import React, { useState } from "react";
import Navbar from "./components/Navbar";
import TagSelector from "./components/TagSelector";
import RecommendationResults from "./components/RecommendationResults";
import "./app.css";

export default function App() {
  const [stage, setStage] = useState("choice"); // 'choice' | 'tags'
  const [category, setCategory] = useState(null);
  const [selectedTags, setSelectedTags] = useState([]);

  function handleChoose(cat) {
    setCategory(cat);
    setStage("tags");
  }

  function handleBack() {
    setCategory(null);
    setSelectedTags([]);
    setStage("choice");
  }

  function onTagsChange(tags) {
    setSelectedTags(tags);
  }

  return (
    <div className="three-grid">
      <Navbar />
      {stage === "choice" ? (
        <>
          <div className="background-one">
            <div className="link-container">
              <button
                className="bubbles link-one"
                onClick={() => handleChoose("Movies")}
                aria-label="Choose Movies"
              >
                <span className="text">Movies</span>
              </button>
            </div>
          </div>

          <div className="background-two">
            <div className="link-container">
              <button
                className="bubbles link-two"
                onClick={() => handleChoose("Books")}
                aria-label="Choose Books"
              >
                <span className="text">Books</span>
                <span className="dec-square" aria-hidden="true" />
              </button>
            </div>
          </div>

          <div className="background-three">
            <div className="link-container">
              <button
                className="bubbles link-three"
                onClick={() => handleChoose("Products")}
                aria-label="Choose Products"
              >
                <span className="text">Products</span>
              </button>
            </div>
          </div>
        </>
      ) : (
        <div className="tags-stage" style={{ paddingTop: "90px" }}>
          <div className="card">
            <div className="row header">
              <button className="btn btn-ghost" onClick={handleBack}>
                ← Back
              </button>
              <h2 className="title">{category}</h2>
              <div style={{ width: 64 }} />
            </div>

            <TagSelector
              categoryType={category.toLowerCase()}
              onTagsChange={onTagsChange}
            />
            
            {selectedTags.length > 0 && (
              <RecommendationResults
                dataset={category}
                selectedTags={selectedTags}
                userId={1}
                model="svd"
              />
            )}
            
            {selectedTags.length === 0 && (
              <div className="results-placeholder" style={{ marginTop: "20px" }}>
                <p className="muted">
                  Select some categories above to get personalized recommendations!
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
