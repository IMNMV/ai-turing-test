// FILE: static/script.js
// At the top of script.js
const API_BASE_URL = window.location.hostname === 'localhost'
    ? 'http://localhost:8000'  // Local development
    : 'https://ai-turing-test-production.up.railway.app';

document.addEventListener('DOMContentLoaded', () => {
    const initialSetupDiv = document.getElementById('initial-setup');
    const chatInterfaceDiv = document.getElementById('chat-interface'); // Main div for chat + assessment
    const assessmentAreaDiv = document.getElementById('assessment-area'); // Now inside chatInterfaceDiv
    const finalPageDiv = document.getElementById('final-page');
    const errorMessageArea = document.getElementById('error-message-area');

    const initialForm = document.getElementById('initial-form');
    const initLoadingDiv = document.getElementById('init-loading');

    const messageList = document.getElementById('message-list');
    const userMessageInput = document.getElementById('user-message-input');
    const sendMessageButton = document.getElementById('send-message-button');
    const typingIndicator = document.getElementById('typing-indicator');
    const chatInputContainer = document.getElementById('chat-input-container'); // Div containing text input and send button

    const confidenceSlider = document.getElementById('confidence-slider');
    const confidenceValueSpan = document.getElementById('confidence-value');
    const submitRatingButton = document.getElementById('submit-rating-button');
    const ratingLoadingDiv = document.getElementById('rating-loading');

    const feelsOffCheckbox = document.getElementById('feels-off-checkbox');
    const commentInputArea = document.getElementById('comment-input-area');
    const feelsOffCommentTextarea = document.getElementById('feels-off-comment');
    const submitCommentButton = document.getElementById('submit-comment-button');
    const commentLoadingDiv = document.getElementById('comment-loading');

    const finalDecisionText = document.getElementById('final-decision-text');
    const finalDecisionTimeText = document.getElementById('final-decision-time-text');
    const confidenceTrendData = document.getElementById('confidence-trend-data');
    const newSessionButton = document.getElementById('new-session-button');
    const loadResearcherDataButton = document.getElementById('load-researcher-data-button');
    const researcherDataContent = document.getElementById('researcher-data-content');
    const researcherDataSection = document.getElementById('researcher-data-section');


    let sessionId = localStorage.getItem('sessionId');
    let currentTurn = 0;
    let aiResponseTimestamp = null;

    // --- Helper Functions ---
    function showError(message) {
        errorMessageArea.textContent = message;
        errorMessageArea.style.display = 'block';
        setTimeout(() => {
            errorMessageArea.style.display = 'none';
        }, 5000);
    }

    // Simplified showPhase for main application states
    function showMainPhase(phase) {
        console.log(`Showing main phase: ${phase}`);
        initialSetupDiv.style.display = 'none';
        chatInterfaceDiv.style.display = 'none';
        finalPageDiv.style.display = 'none';

        if (phase === 'initial') initialSetupDiv.style.display = 'block';
        else if (phase === 'chat_and_assessment_flow') chatInterfaceDiv.style.display = 'block'; // Renamed for clarity
        else if (phase === 'final') finalPageDiv.style.display = 'block';
    }


    function addMessageToUI(text, sender) {
        const messageBubble = document.createElement('div');
        messageBubble.classList.add('message-bubble', sender);
        messageBubble.textContent = text;
        messageList.appendChild(messageBubble);
        messageList.scrollTop = messageList.scrollHeight;
        console.log(`Added message from ${sender}. messageList.childElementCount: ${messageList.childElementCount}`);
    }

    // --- Event Listeners ---
    initialForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        initLoadingDiv.style.display = 'block';
        initialForm.querySelector('button').disabled = true;

        const formData = new FormData(initialForm);

        
        // Handle checkboxes for multi-select fields
        const ai_models_used = [];
        const ethnicity = [];
        
        formData.getAll('ai_models_used').forEach(model => ai_models_used.push(model));
        formData.getAll('ethnicity').forEach(eth => ethnicity.push(eth));
        
        const data = {
            // AI Experience
            ai_usage_frequency: parseInt(formData.get('ai_usage_frequency')),
            ai_models_used: ai_models_used,
            ai_detection_confidence_self: parseInt(formData.get('ai_detection_confidence_self')),
            ai_detection_confidence_others: parseInt(formData.get('ai_detection_confidence_others')),
            ai_folk_understanding: formData.get('ai_folk_understanding'),
            
            // Conversation Style
            conversation_style: parseInt(formData.get('conversation_style')),
            risk_preference: parseInt(formData.get('risk_preference')),
            trust_in_ai: parseInt(formData.get('trust_in_ai')),
            
            // Demographics
            age: parseInt(formData.get('age')),
            gender: formData.get('gender'),
            education: formData.get('education'),
            ethnicity: ethnicity,
            income: formData.get('income')
        };

        try {
            const response = await fetch(`${API_BASE_URL}/initialize_study`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
            const result = await response.json();
            if (response.ok) {
                sessionId = result.session_id;
                localStorage.setItem('sessionId', sessionId);
                currentTurn = 0;
                messageList.innerHTML = '';
                showMainPhase('chat_and_assessment_flow'); // Show the chat interface
                assessmentAreaDiv.style.display = 'none';    // Ensure assessment is hidden initially
                chatInputContainer.style.display = 'flex';  // Show chat input
                userMessageInput.disabled = false;
                sendMessageButton.disabled = false;
                userMessageInput.focus();
            } else {
                showError(result.detail || 'Failed to initialize study.');
            }
        } catch (error) {
            showError('Error initializing study: ' + error.message);
        } finally {
            initLoadingDiv.style.display = 'none';
            initialForm.querySelector('button').disabled = false;
        }
    });

    sendMessageButton.addEventListener('click', handleSendMessage);
    userMessageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSendMessage();
        }
    });

    // SLIDER VALUE UPDATE - FIX
    confidenceSlider.addEventListener('input', () => {
        confidenceValueSpan.textContent = parseFloat(confidenceSlider.value).toFixed(2);
    });

    async function handleSendMessage() {
        const messageText = userMessageInput.value.trim();
        if (!messageText || !sessionId) return;

        // Chat interface should already be visible
        addMessageToUI(messageText, 'user');

        userMessageInput.value = '';
        userMessageInput.disabled = true;
        sendMessageButton.disabled = true;
        // typingIndicator.style.display = 'block'; // Old: Show indicator immediately

        // Hide chat input AND assessment area while AI is (potentially) typing
        chatInputContainer.style.display = 'none';
        assessmentAreaDiv.style.display = 'none';

        // --- NEW: Delayed Typing Indicator ---
        // Show typing indicator after a random delay (e.g., 2-5 seconds as in paper)
        // The paper says "2+U(0,3)s", which means 2 seconds + a random uniform value between 0 and 3 seconds.
        // So, a delay between 2 and 5 seconds.
        const indicatorDelay = Math.random() * (5000 - 2000) + 2000; // milliseconds
        console.log(`Setting timeout to show typing indicator in ${indicatorDelay / 1000} seconds.`);
        setTimeout(() => {
            // Only show if we haven't already received a response (unlikely but a good check)
            // and if the assessment area isn't already visible (meaning AI responded super fast)
            if (assessmentAreaDiv.style.display === 'none' && chatInputContainer.style.display === 'none') {
                 typingIndicator.style.display = 'block';
                 console.log("Typing indicator shown after delay.");
            }
        }, indicatorDelay);
        // --- END NEW ---

        console.log('Sending message to server:', { session_id: sessionId, message: messageText });

        try {
            // Fetch call to the backend happens immediately after setting the timeout for the indicator
            const response = await fetch(`${API_BASE_URL}/send_message`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, message: messageText }),
            });
            const result = await response.json();
            console.log('Received response from /send_message:', result);

            // Whether the indicator was shown or not, hide it now that we have a response (or error)
            typingIndicator.style.display = 'none';

            if (response.ok) {
                addMessageToUI(result.ai_response, 'assistant');
                currentTurn = result.turn;
                aiResponseTimestamp = result.timestamp;

                // AI has responded, now show the assessment area
                // Keep chat input hidden until assessment is done
                assessmentAreaDiv.style.display = 'block';
                chatInputContainer.style.display = 'none'; // Keep chat input hidden

                assessmentAreaDiv.querySelector('h4').textContent = "Your Assessment"; // h4 now
                confidenceSlider.value = 0.5;
                confidenceValueSpan.textContent = '0.50'; // Update span as well
                confidenceSlider.disabled = false;
                submitRatingButton.style.display = 'block';
                submitRatingButton.disabled = false;
                feelsOffCheckbox.checked = false;
                commentInputArea.style.display = 'none';
                feelsOffCommentTextarea.value = '';
                //document.getElementById('comment-section').style.display = 'block';
                messageList.scrollTop = messageList.scrollHeight; // Scroll to see latest AI message and assessment

            } else {
                showError(result.detail || 'Failed to send message or get AI response.');
                addMessageToUI("Sorry, I couldn't process that. Please try again.", 'assistant');
                // If error, re-enable chat input
                userMessageInput.disabled = false;
                sendMessageButton.disabled = false;
                chatInputContainer.style.display = 'flex'; // Show chat input again
                assessmentAreaDiv.style.display = 'none';   // Keep assessment hidden
            }
        } catch (error) {
            showError('Error sending message: ' + error.message);
            console.error('Catch block error in handleSendMessage:', error);
            typingIndicator.style.display = 'none'; // Ensure it's hidden on error
            addMessageToUI("Sorry, a technical error occurred. Please try again.", 'assistant');
            userMessageInput.disabled = false;
            sendMessageButton.disabled = false;
            chatInputContainer.style.display = 'flex';
            assessmentAreaDiv.style.display = 'none';
        }
    }

    submitRatingButton.addEventListener('click', async () => {
        if (!sessionId) return;

        // --- VALIDATION FOR COMMENT INTENT ---
        if (feelsOffCheckbox.checked && feelsOffCommentTextarea.value.trim() === '') {
            showError("Please provide a comment if 'Include a comment' is checked, or uncheck the box to proceed with rating only.");
            return; // Stop further execution
        }
        // --- END VALIDATION ---

        ratingLoadingDiv.style.display = 'block';
        submitRatingButton.disabled = true;
        confidenceSlider.disabled = true;

        const confidence = parseFloat(confidenceSlider.value);
        let decisionTimeSeconds = null;
        if (aiResponseTimestamp) {
            decisionTimeSeconds = (new Date().getTime() / 1000) - aiResponseTimestamp;
        } else {
            console.warn("aiResponseTimestamp was not set, decision time will be null.");
        }

        console.log('Submitting rating. Confidence:', confidence, 'Decision Time:', decisionTimeSeconds);

        try {
            const response = await fetch(`${API_BASE_URL}/submit_rating`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, confidence: confidence, decision_time_seconds: decisionTimeSeconds }),
            });
            const result = await response.json();
            console.log('Rating submission response from server:', result);

            if (response.ok) {
                if (result.study_over) {
                    showMainPhase('final');
                    displayFinalPage(result.session_data_summary);
                } else {
                    // Study continues. Rating has been successfully submitted.
                    // Now, determine UI based on comment intent.

                    if (feelsOffCheckbox.checked && feelsOffCommentTextarea.value.trim() !== '') {
                        assessmentAreaDiv.querySelector('h4').textContent = "Rating Submitted. Now, please submit your comment:";
                        submitRatingButton.style.display = 'none';
                        confidenceSlider.style.display = 'none';
                        confidenceValueSpan.style.display = 'none';
                        document.getElementById('submit-rating-button').previousElementSibling.style.display = 'none'; // Hide the label for confidence slider

                        document.getElementById('comment-section').style.display = 'block';
                        if (!commentInputArea.style.display || commentInputArea.style.display === 'none') {
                            commentInputArea.style.display = 'block';
                        }
                        feelsOffCommentTextarea.focus();
                        chatInputContainer.style.display = 'none';
                    } else {
                        assessmentAreaDiv.style.display = 'none';
                        chatInputContainer.style.display = 'flex';
                        userMessageInput.disabled = false;
                        sendMessageButton.disabled = false;
                        userMessageInput.focus();
                    }
                }
            } else { // fetch response not OK
                showError(result.detail || 'Failed to submit rating.');
                submitRatingButton.disabled = false;
                confidenceSlider.disabled = false;
            }
        } catch (error) { // Network error or JSON parsing error
            showError('Error submitting rating: ' + error.message);
            console.error('Catch block error in submitRatingButton:', error);
            submitRatingButton.disabled = false;
            confidenceSlider.disabled = false;
        } finally {
            ratingLoadingDiv.style.display = 'none';
        }
    });

    submitCommentButton.addEventListener('click', async () => {
        if (!sessionId) {
            showError("Session ID is missing. Please refresh.");
            return;
        }
        if (!feelsOffCheckbox.checked) {
            showError("Please check 'Include a comment' to submit a comment.");
            return;
        }

        const commentText = feelsOffCommentTextarea.value.trim();
        if (commentText === '') {
            showError("Comment cannot be empty if 'Include a comment' is checked.");
            return;
        }

        commentLoadingDiv.style.display = 'block';
        submitCommentButton.disabled = true;
        feelsOffCheckbox.disabled = true;

        try {
            const response = await fetch(`${API_BASE_URL}/submit_comment`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, comment: commentText }),
            });
            const result = await response.json();
            if (response.ok) {
                console.log("Comment submitted successfully.");
                feelsOffCommentTextarea.value = '';
                feelsOffCheckbox.checked = false;
                commentInputArea.style.display = 'none';

                if (finalPageDiv.style.display === 'none') { // Only if study is not over
                    assessmentAreaDiv.style.display = 'none';
                    chatInputContainer.style.display = 'flex';
                    userMessageInput.disabled = false;
                    sendMessageButton.disabled = false;
                    userMessageInput.focus();
                }
            } else {
                showError(result.detail || 'Failed to submit comment. Please try again.');
            }
        } catch (error) {
            showError('Error submitting comment: ' + error.message);
            console.error('Catch block error in submitCommentButton:', error);
        } finally {
            commentLoadingDiv.style.display = 'none';
            if (commentInputArea.style.display !== 'none') {
                 submitCommentButton.disabled = false;
                 feelsOffCheckbox.disabled = false;
            }
        }
    });


    feelsOffCheckbox.addEventListener('change', () => {
        commentInputArea.style.display = feelsOffCheckbox.checked ? 'block' : 'none';
    });



    function displayFinalPage(summary) {
        finalDecisionText.textContent = `You determined that you were ${summary.ai_detected ? 'talking to an AI' : 'talking to a human'}.`;
        if (summary.final_decision_time) {
            finalDecisionTimeText.textContent = `Time taken to make final decision: ${parseFloat(summary.final_decision_time).toFixed(2)} seconds.`;
        }
        let trendText = "Turn | Confidence | Decision Time (s)\n";
        trendText += "---------------------------------------\n";
        if (summary.confidence_ratings) {
            summary.confidence_ratings.forEach(r => {
                trendText += `${String(r.turn).padEnd(4)} | ${parseFloat(r.confidence).toFixed(2).padEnd(10)} | ${r.decision_time_seconds ? parseFloat(r.decision_time_seconds).toFixed(2) : 'N/A'}\n`;
            });
        }
        confidenceTrendData.textContent = trendText;
        researcherDataSection.style.display = 'block';
        researcherDataContent.textContent = "";
    }

    newSessionButton.addEventListener('click', () => {
        localStorage.removeItem('sessionId');
        sessionId = null;
        currentTurn = 0;
        aiResponseTimestamp = null;
        messageList.innerHTML = '';
        initialForm.reset();
        researcherDataSection.style.display = 'none';
        researcherDataContent.textContent = '';
        showMainPhase('initial');
    });

    loadResearcherDataButton.addEventListener('click', async () => {
        if (!sessionId) {
            researcherDataContent.textContent = "No active session ID to load data for.";
            return;
        }
        loadResearcherDataButton.disabled = true;
        researcherDataContent.textContent = "Loading researcher data...";
        try {
            const response = await fetch(`${API_BASE_URL}/get_researcher_data/${sessionId}`);
            if (!response.ok) {
                const errorText = await response.text();
                researcherDataContent.textContent = `Error loading researcher data: ${response.status} ${response.statusText}. Details: ${errorText}`;
                return;
            }
            const data = await response.json();
            researcherDataContent.textContent = JSON.stringify(data, null, 2);
        } catch (error) {
            researcherDataContent.textContent = `Error fetching or parsing researcher data: ${error.message}`;
            console.error("Error in loadResearcherDataButton:", error)
        } finally {
            loadResearcherDataButton.disabled = false;
        }
    });

    // --- Initial Page Load ---
    localStorage.removeItem('sessionId'); // Clear session on fresh load
    aiResponseTimestamp = null;
    showMainPhase('initial');
});
