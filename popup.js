const testingResult = `
<!DOCTYPE html>
<html>

<head>
    <meta charset="UTF-8">
    <title>Detect phishing API</title>
    <link href='https://fonts.googleapis.com/css?family=Pacifico' rel='stylesheet' type='text/css'>
    <link href='https://fonts.googleapis.com/css?family=Arimo' rel='stylesheet' type='text/css'>
    <link href='https://fonts.googleapis.com/css?family=Hind:300' rel='stylesheet' type='text/css'>
    <link href='https://fonts.googleapis.com/css?family=Open+Sans+Condensed:300' rel='stylesheet' type='text/css'>
</head>

<body>
    <div class="login">
        <h1>Predict if email is phishing</h1>

        <!-- Main Input For Receiving Query to our ML -->
        <form action="/" method="post">
            <input type="text" name="experience" placeholder="File name please:" required="required" />
            <button type="submit" class="btn btn-primary btn-block btn-large">Predict</button>
        </form>

        <br>
        <br>
    Model: Classifiers/SGDClassifier.joblib does not support probability estimates
Nonetheless, the prediction is: Phishing Email
Model: Classifiers/DecisionTreeClassifier.joblib
Prediction: Safe Email, Certainty: 1.0
Model: Classifiers/LogisticRegression.joblib
Prediction: Phishing Email, Certainty: 0.9594101527100533
Model: Classifiers/RandomForestClassifier.joblib
Prediction: Phishing Email, Certainty: 0.6
Model: Classifiers/AdaBoostClassifier.joblib
Prediction: Phishing Email, Certainty: 0.5126577472528376

    </div>
</body>

</html>
`;

// Style  / Animations //
const keyframes = `
  @keyframes bg1 {
    0% {
      opacity: 0;
      transform: translateX(20px);
    }
    100% {
      opacity: 1;
      transform: translateX(0);
    }
  }
  @keyframes bg2 {
    0% {
      opacity : 0;
    }
    100% {
      opacity : 1;
    }
  }
  @keyframes bg3 {
    0% {
      opacity : 1;
    }
    100% {
      opacity : 0;
    }
  }
`;

const styleElement = document.createElement("style");
styleElement.innerHTML = keyframes;
document.head.appendChild(styleElement);

// Style //

document.addEventListener("DOMContentLoaded", function () {
  var readButton = document.getElementById("readButton");
  readButton.addEventListener("click", function () {
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      chrome.tabs.sendMessage(tabs[0].id, { action: "readText" });
    });
  });
});

let found = false;
// Function to observe changes in the DOM
function observeDOM() {
  // Select the target node
  var targetNode = document.body;

  // Options for the observer (check for additions to the DOM)
  var config = { childList: true, subtree: true };

  // Callback function to execute when mutations are observed
  var callback = function (mutationsList, observer) {
    for (var mutation of mutationsList) {
      if (mutation.type === "childList") {
        // Check if the target node or its subtree has added nodes
        if (mutation.addedNodes.length > 0) {
          // Iterate through added nodes to check if the target element is added
          mutation.addedNodes.forEach(function (node) {
            var sortContainer = document.querySelector("td.c2");
            var phishingButton = document.querySelector(".phishing");

            if (!phishingButton && found) found = false;

            if (node.nodeType === 1 && sortContainer && !found) {
              found = true;
              addButtonToInterface();
            }
          });
        }
      }
    }
  };

  // Create a MutationObserver instance
  var observer = new MutationObserver(callback);
  // Start observing the target node for configured mutations
  observer.observe(targetNode, config);
}

// Function to add your button
function addButtonToInterface() {
  // Create your button element
  var sortButton = document.createElement("button");
  sortButton.textContent = "Phishing Risk?";
  // Apply styles
  sortButton.style.padding = "4px 10px";
  sortButton.style.fontSize = "smaller";
  sortButton.style.marginLeft = "18px";
  sortButton.style.backgroundColor = "rgb(235, 236, 237)";
  sortButton.style.color = "#4e4e4e";
  sortButton.style.border = "none";
  sortButton.style.cursor = "pointer";
  sortButton.style.borderRadius = "0px";
  sortButton.classList.add("phishing");

  // Create your button element
  var resultButton = document.createElement("button");
  resultButton.id = "resultButton";
  // Apply styles
  resultButton.style.padding = "4px 10px";
  resultButton.style.fontSize = "smaller";
  resultButton.style.backgroundColor = "rgb(77, 77, 77)";
  resultButton.style.color = "rgb(235, 236, 237)";
  resultButton.style.border = "none";
  resultButton.style.borderRadius = "0px";

  resultButton.classList.add("phishing");

  // Find the target element to append your button
  var sortContainer = document.querySelector("td.c2");

  // Check if the target element exists
  if (sortContainer) {
    // Append the button to the target element
    sortContainer.appendChild(sortButton);

    // sample api call to be replaced //
    sortButton.addEventListener("click", () => {
      var sortButton2 = document.createElement("button");
      sortButton2.style.border = "none";
      sortButton2.style.borderRadius = "0px";
      sortButton2.style.backgroundColor = "#00000000";
      sortButton2.style.fontSize = "smaller";
      sortButton2.style.opacity = "0";
      sortButton2.style.animation =
        "bg2 1s 0s cubic-bezier(0.6, 0.1, 0.165, 1), bg3 1s 1s cubic-bezier(0.6, 0.1, 0.165, 1)";
      sortButton2.style.animationFillMode = "forwards";
      sortButton2.textContent = "loading...";
      sortContainer.appendChild(sortButton2);

      // temp fake request
      // setTimeout(() => {
      //   // temp fake request
      //   fetch("https://randomuser.me/api/")
      //     .then((response) => {
      //       if (!response.ok) {
      //         throw new Error("Network response was not ok");
      //       }
      //       return response.json();
      //     })
      //     .then((data) => {
      //       sortButton2.style.display = "None";
      //       sortContainer.appendChild(resultButton);

      //       const buttonElement = document.getElementById("resultButton");
      //       buttonElement.style.opacity = "0";
      //       buttonElement.style.transform = "translateX(20px)";
      //       buttonElement.style.animation =
      //         "bg1 2.4s 0s cubic-bezier(0.6, 0.1, 0.165, 1)";
      //       buttonElement.style.animationFillMode = "forwards";

      //       let parsed = parser(testingResult);
      //       let consensus = classifyEmail(parsed);
      //       console.log(consensus);
      //       if (consensus.classification == "Danger") {
      //         resultButton.textContent =
      //           "High, Risk estimate : " +
      //           consensus.averageCertainty.toPrecision(2) +
      //           "%";
      //         resultButton.style.backgroundColor = "rgb(242,28,28)";
      //       }
      //       if (consensus.classification == "Moderate") {
      //         resultButton.style.backgroundColor = "rgb(242,138,28)";
      //         resultButton.textContent =
      //           "Moderate, Risk estimate : " +
      //           consensus.averageCertainty.toPrecision(2) +
      //           "%";
      //       }
      //       if (consensus.classification == "Safe") {
      //         resultButton.textContent =
      //           "Low, Safety estimate : " +
      //           consensus.averageCertainty.toPrecision(2) +
      //           "%";
      //         resultButton.style.backgroundColor = "rgb(7,138,68)";
      //       }
      //     })
      //     .catch((error) => {
      //       console.error("There was a problem with the API request:", error);
      //     });
      // }, 2500);

      // Actual request which should work once the api runs //

      const formData = new FormData();
      formData.append("experience", document.querySelector(".a3s").textContent);

      fetch("https://3.14.250.99/", {
        method: "POST",
        body: formData,
      })
        .then((response) => {
          console.log(response);
          if (!response.ok) {
            throw new Error("Network response was not ok");
          }
          return response.text();
        })
        .then((data) => {
          sortContainer.appendChild(resultButton);
          const keyframes = `
          @keyframes bg1 {
            0% {
              opacity: 0;
              transform: translateX(20px);
            }
            100% {
              opacity: 1;
              transform: translateX(0);
            }
          }
        `;

          const styleElement = document.createElement("style");
          styleElement.innerHTML = keyframes;
          document.head.appendChild(styleElement);

          // Now you can apply the styles as before
          const buttonElement = document.getElementById("resultButton");
          buttonElement.style.opacity = "0";
          buttonElement.style.transform = "translateX(20px)";
          buttonElement.style.animation =
            "bg1 2.4s 0s cubic-bezier(0.6, 0.1, 0.165, 1)";
          buttonElement.style.animationFillMode = "forwards";

          let parsed = parser(data);
          let consensus = classifyEmail(parsed);
          console.log(consensus);
          if (consensus.classification == "Danger") {
            resultButton.textContent =
              "High, Risk estimate : " +
              consensus.averageCertainty.toPrecision(2) +
              "%";
            resultButton.style.backgroundColor = "rgb(242,28,28)";
          }
          if (consensus.classification == "Moderate") {
            resultButton.style.backgroundColor = "rgb(242,156,28)";
            resultButton.textContent =
              "Moderate, Risk estimate : " +
              consensus.averageCertainty.toPrecision(2) +
              "%";
          }
          if (consensus.classification == "Safe") {
            resultButton.textContent =
              "Low, Safety estimate : " +
              consensus.averageCertainty.toPrecision(2) +
              "%";
            resultButton.style.backgroundColor = "rgb(7,138,68)";
          }
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    });
  } else {
    // If the target element doesn't exist yet, keep observing the DOM
    observeDOM();
  }
}
observeDOM();

//Parse the result from the api call, extracting only the models
const parser = (htmlString) => {
  const parser = new DOMParser();
  const doc = parser.parseFromString(htmlString, "text/html");
  const loginDiv = doc.querySelector(".login");
  let result = loginDiv.textContent.trim();
  let cleanResult = result
    .split("\n")
    .slice(10)
    .map((k) => k.trim());
  return cleanResult;
};

function classifyEmail(predictions) {
  let phishingCount = 0;
  let phishingCountWithCertainty = 0;
  let safeCountwithCertainty = 0;
  let safeCertaintySum = 0;
  let phishingCertaintySum = 0;

  for (let i = 1; i < predictions.length; i += 1) {
    console.log(predictions[i]);
    if (predictions[i].includes("Phishing Email")) {
      const certaintyLine = predictions[i];
      const certaintyMatch = certaintyLine.match(/Certainty: ([0-9.]+)/);
      phishingCount++;
      if (certaintyMatch && certaintyMatch[1]) {
        phishingCountWithCertainty++;
        const certainty = parseFloat(certaintyMatch[1]);
        phishingCertaintySum += certainty;
      }
    }
    if (predictions[i].includes("Safe Email")) {
      const certaintyLine = predictions[i];
      const certaintyMatch = certaintyLine.match(/Certainty: ([0-9.]+)/);
      if (certaintyMatch && certaintyMatch[1]) {
        safeCountwithCertainty++;
        const certainty = parseFloat(certaintyMatch[1]);
        safeCertaintySum += certainty;
      }
    }
  }
  console.log(phishingCount);
  if (phishingCount == 5) {
    const averageCertainty =
      (phishingCertaintySum / phishingCountWithCertainty) * 100;
    return { classification: "Danger", averageCertainty };
  } else if (phishingCount >= 3) {
    const averageCertainty =
      (phishingCertaintySum / phishingCountWithCertainty) * 100;
    return { classification: "Moderate", averageCertainty };
  } else {
    const averageCertainty = (safeCertaintySum / safeCountwithCertainty) * 100;
    return { classification: "Safe", averageCertainty: averageCertainty };
  }
}
