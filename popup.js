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
  var callback = function(mutationsList, observer) {
      for(var mutation of mutationsList) {
          if (mutation.type === 'childList') {
              // Check if the target node or its subtree has added nodes
              if (mutation.addedNodes.length > 0) {
                  // Iterate through added nodes to check if the target element is added
                  mutation.addedNodes.forEach(function(node) {
                    var sortContainer = document.querySelector('td.c2');
                    var phishingButton = document.querySelector('.phishing');

                    if(!phishingButton && found ) found = false;
                    
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
  console.log("called!!!!!")
  var sortButton = document.createElement('button');
  sortButton.textContent = 'Check for Phishing?';
  sortButton.id = 'sortButton';

  // Apply styles
  sortButton.style.padding = '4px 10px';
  sortButton.style.fontSize = 'smaller';
  sortButton.style.marginLeft = '18px';
  sortButton.style.backgroundColor = 'rgb(235, 236, 237)';
  sortButton.style.color = '#4e4e4e';
  sortButton.style.border = 'none';
  sortButton.style.borderRadius = '0px';
  sortButton.style.cursor = 'crosshair';
  sortButton.classList.add('phishing');

  

  // Find the target element to append your button
  var sortContainer = document.querySelector('td.c2');
  // Check if the target element exists
  if (sortContainer) {
      // Append the button to the target element
      sortContainer.appendChild(sortButton);
      sortButton.addEventListener('click', () => {
        const formData = new FormData();
        formData.append('experience', document.querySelector('.a3s').textContent);
        // POST request with fake form data
        fetch('https://3.14.250.99/', {
          method: 'POST',
          body: formData
        })
        .then(response => {
          if (!response.ok) {
            throw new Error('Network response was not ok');
          }
          return response.json();
        })
        .then(data => {
          console.log('Success:', data);
        })
        .catch(error => {
          console.error('Error:', error);
        });
    });
  } else {
      // If the target element doesn't exist yet, keep observing the DOM
      observeDOM();
  }
}

observeDOM();