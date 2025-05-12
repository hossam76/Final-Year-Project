/**
 * Advanced JavaScript functionality for Fake News Detection System
 * Final Year Project - Enhanced UI
 */

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Initialize theme preference
    initTheme();
    
    // Setup event listeners
    setupEventListeners();
    
    // Initialize animations
    initAnimations();
    
    // Add loading state to form submissions
    initFormLoadingState();
    
    // Initialize charts if needed
    if (document.getElementById('modelComparisonChart')) {
        initModelComparisonChart();
    }
    
    if (document.getElementById('explanationChart')) {
        initExplanationChart();
    }
    
    if (document.getElementById('posChart')) {
        initPOSChart();
    }
    
    if (document.getElementById('wordFrequencyChart')) {
        initWordFrequencyChart();
    }
});

/**
 * Initialize theme based on saved preference or system setting
 */
function initTheme() {
    const savedTheme = localStorage.getItem('theme');
    
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-mode');
        if (document.querySelector('.theme-toggle i')) {
            document.querySelector('.theme-toggle i').classList.replace('fa-moon', 'fa-sun');
        }
    } else if (savedTheme === 'light') {
        document.body.classList.remove('dark-mode');
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        // Use system preference as fallback
        document.body.classList.add('dark-mode');
        if (document.querySelector('.theme-toggle i')) {
            document.querySelector('.theme-toggle i').classList.replace('fa-moon', 'fa-sun');
        }
    }
}

/**
 * Set up event listeners for interactive elements
 */
function setupEventListeners() {
    // Theme toggle functionality
    const themeToggle = document.querySelector('.theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            if (document.body.classList.contains('dark-mode')) {
                document.body.classList.remove('dark-mode');
                localStorage.setItem('theme', 'light');
                this.querySelector('i').classList.replace('fa-sun', 'fa-moon');
            } else {
                document.body.classList.add('dark-mode');
                localStorage.setItem('theme', 'dark');
                this.querySelector('i').classList.replace('fa-moon', 'fa-sun');
            }
        });
    }
    
    // Modern tabs functionality
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tabId = this.getAttribute('data-tab');
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.modern-tab-pane').forEach(pane => pane.classList.remove('active'));
            
            // Add active class to current tab and pane
            this.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        });
    });
}

/**
 * Initialize animations for UI elements
 */
function initAnimations() {
    // Animated counters
    const counters = document.querySelectorAll('.counter');
    
    if (counters.length > 0) {
        counters.forEach(counter => {
            const target = parseFloat(counter.getAttribute('data-target'));
            const duration = 2000; // ms
            const step = 60; // updates per second
            const increment = target / (duration / (1000 / step));
            
            let current = 0;
            const updateCounter = setInterval(() => {
                current += increment;
                
                if (current >= target) {
                    counter.textContent = target < 1 ? target.toFixed(1) : Math.round(target);
                    clearInterval(updateCounter);
                } else {
                    counter.textContent = current < 1 ? current.toFixed(1) : Math.round(current);
                }
            }, 1000 / step);
        });
    }
    
    // Enhance card hover effects
    const cards = document.querySelectorAll('.neo-card, .tech-card, .model-result-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.boxShadow = '0 10px 30px rgba(0,0,0,0.15)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = '';
            this.style.boxShadow = '';
        });
    });
}

/**
 * Add loading state to form submissions
 */
function initFormLoadingState() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                const originalContent = submitBtn.innerHTML;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
                submitBtn.disabled = true;
                
                // Create a loading overlay
                const overlay = document.createElement('div');
                overlay.className = 'loading-overlay';
                overlay.innerHTML = `
                    <div class="spinner"></div>
                    <p class="mt-3">Analyzing content...</p>
                `;
                document.body.appendChild(overlay);
                
                // Restore button state in case of error
                setTimeout(() => {
                    if (document.querySelector('.loading-overlay')) {
                        submitBtn.innerHTML = originalContent;
                        submitBtn.disabled = false;
                        document.querySelector('.loading-overlay').remove();
                    }
                }, 30000); // 30 second timeout
            }
        });
    });
}

/**
 * Initialize model comparison chart
 */
function initModelComparisonChart() {
    const ctx = document.getElementById('modelComparisonChart').getContext('2d');
    
    // Get data from the results grid
    const models = [];
    const confidences = [];
    const colors = [];
    
    document.querySelectorAll('.model-result-card').forEach(card => {
        const model = card.querySelector('.model-result-header span').textContent;
        const confidence = parseFloat(card.querySelector('.confidence-value').textContent);
        const isPredictionReal = card.querySelector('.prediction').classList.contains('prediction-real');
        
        models.push(model);
        confidences.push(confidence);
        colors.push(isPredictionReal ? 'rgba(40, 167, 69, 0.7)' : 'rgba(220, 53, 69, 0.7)');
    });
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models,
            datasets: [{
                label: 'Confidence (%)',
                data: confidences,
                backgroundColor: colors,
                borderColor: colors.map(color => color.replace('0.7', '1')),
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Confidence: ${context.parsed.y.toFixed(2)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        display: true,
                        drawBorder: false,
                        color: 'rgba(200, 200, 200, 0.15)'
                    },
                    ticks: {
                        font: {
                            family: "'Poppins', sans-serif"
                        }
                    }
                },
                x: {
                    grid: {
                        display: false,
                        drawBorder: false
                    },
                    ticks: {
                        font: {
                            family: "'Poppins', sans-serif"
                        }
                    }
                }
            },
            animation: {
                duration: 2000,
                easing: 'easeOutQuart'
            }
        }
    });
}

/**
 * Initialize word frequency chart
 */
function initWordFrequencyChart() {
    const ctx = document.getElementById('wordFrequencyChart').getContext('2d');
    
    // Extract data from text analysis if available
    const textAnalysis = window.textAnalysis || {};
    const words = Object.keys(textAnalysis.top_words || {}).slice(0, 5);
    const frequencies = words.map(word => textAnalysis.top_words[word]);
    
    // Use placeholder data if no data is available
    const labels = words.length ? words : ['Sample', 'Example', 'Word', 'Test', 'Demo'];
    const data = frequencies.length ? frequencies : [24, 18, 16, 15, 12];
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Frequency',
                data: data,
                backgroundColor: 'rgba(67, 97, 238, 0.7)',
                borderColor: 'rgba(67, 97, 238, 1)',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    grid: {
                        display: true,
                        drawBorder: false,
                        color: 'rgba(200, 200, 200, 0.15)'
                    }
                },
                y: {
                    grid: {
                        display: false,
                        drawBorder: false
                    }
                }
            },
            animation: {
                duration: 1500,
                easing: 'easeOutQuart'
            }
        }
    });
}

/**
 * Initialize POS chart
 */
function initPOSChart() {
    const ctx = document.getElementById('posChart').getContext('2d');
    
    // Extract data from text analysis if available
    const textAnalysis = window.textAnalysis || {};
    const posLabels = {
        'NOUN': 'Nouns',
        'VERB': 'Verbs',
        'ADJ': 'Adjectives',
        'ADV': 'Adverbs',
        'PRON': 'Pronouns',
        'DET': 'Determiners',
        'ADP': 'Adpositions',
        'CONJ': 'Conjunctions',
        'PRT': 'Particles'
    };
    
    const posCounts = textAnalysis.pos_counts || {};
    const labels = Object.keys(posCounts).map(pos => posLabels[pos] || pos);
    const counts = Object.values(posCounts);
    
    // Use placeholder data if no data is available
    const chartLabels = labels.length ? labels : ['Nouns', 'Verbs', 'Adjectives', 'Adverbs', 'Others'];
    const chartData = counts.length ? counts : [45, 28, 15, 12, 10];
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: chartLabels,
            datasets: [{
                data: chartData,
                backgroundColor: [
                    'rgba(67, 97, 238, 0.7)',
                    'rgba(77, 201, 240, 0.7)',
                    'rgba(114, 9, 183, 0.7)',
                    'rgba(6, 214, 160, 0.7)',
                    'rgba(255, 209, 102, 0.7)',
                    'rgba(239, 71, 111, 0.7)',
                    'rgba(17, 138, 178, 0.7)',
                    'rgba(7, 59, 76, 0.7)',
                    'rgba(6, 123, 194, 0.7)'
                ],
                borderColor: [
                    'rgba(67, 97, 238, 1)',
                    'rgba(77, 201, 240, 1)',
                    'rgba(114, 9, 183, 1)',
                    'rgba(6, 214, 160, 1)',
                    'rgba(255, 209, 102, 1)',
                    'rgba(239, 71, 111, 1)',
                    'rgba(17, 138, 178, 1)',
                    'rgba(7, 59, 76, 1)',
                    'rgba(6, 123, 194, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 11
                        },
                        boxWidth: 15
                    }
                }
            },
            animation: {
                animateRotate: true,
                animateScale: true,
                duration: 2000,
                easing: 'easeOutQuart'
            },
            cutout: '65%'
        }
    });
}

/**
 * Initialize explanation chart for AI explainability
 */
function initExplanationChart() {
    const ctx = document.getElementById('explanationChart').getContext('2d');
    
    // Extract data from explanation if available
    const explanation = window.explanation || {};
    const features = explanation.explanation || [];
    
    const words = features.map(item => item.word);
    const weights = features.map(item => item.weight);
    
    // Use placeholder data if no data is available
    const labels = words.length ? words : ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'];
    const data = weights.length ? weights : [0.4, 0.3, -0.2, -0.5, 0.2];
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Feature Impact',
                data: data,
                backgroundColor: data.map(value => value >= 0 ? 'rgba(40, 167, 69, 0.7)' : 'rgba(220, 53, 69, 0.7)'),
                borderColor: data.map(value => value >= 0 ? 'rgba(40, 167, 69, 1)' : 'rgba(220, 53, 69, 1)'),
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.raw;
                            return `Impact: ${value.toFixed(4)} (${value >= 0 ? 'Real' : 'Fake'})`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    grid: {
                        display: true,
                        drawBorder: false,
                        color: 'rgba(200, 200, 200, 0.15)'
                    }
                },
                y: {
                    grid: {
                        display: false,
                        drawBorder: false
                    }
                }
            },
            animation: {
                duration: 1500,
                easing: 'easeOutQuart'
            }
        }
    });
}

/**
 * FakeNewsGuard - Advanced Visualizations
 * JavaScript file for handling advanced data visualizations and interactions
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    if (tooltipTriggerList.length > 0) {
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl)
        });
    }
    
    // Initialize circular progress bars
    initProgressCircles();
    
    // Load Chart.js if available
    if (typeof Chart !== 'undefined') {
        initCharts();
    }
});

/**
 * Initialize progress circles for language analysis indicators
 */
function initProgressCircles() {
    document.querySelectorAll('.progress-circle').forEach(function(circle) {
        if (!circle) return;
        
        const percentageAttr = circle.getAttribute('data-percentage');
        if (!percentageAttr) return;
        
        const percentage = parseInt(percentageAttr);
        const degrees = (percentage / 100) * 360;
        const isHalf = degrees > 180;
        
        const rightBar = circle.querySelector('.progress-circle-right .progress-circle-bar');
        const leftBar = circle.querySelector('.progress-circle-left .progress-circle-bar');
        
        if (!rightBar || !leftBar) return;
        
        try {
            if (isHalf) {
                rightBar.style.transform = 'rotate(180deg)';
                leftBar.style.transform = `rotate(${degrees - 180}deg)`;
            } else {
                rightBar.style.transform = `rotate(${degrees}deg)`;
            }
            
            // Set color based on type
            let color = '#17a2b8'; // Default blue
            
            const indicatorCard = circle.closest('.indicator-card');
            if (!indicatorCard) return;
            
            const heading = indicatorCard.querySelector('h6');
            if (!heading) return;
            
            const headingText = heading.textContent;
            
            if (headingText.includes('Emotional')) {
                color = percentage > 30 ? '#dc3545' : '#6c757d'; // Red if high emotional content
            } else if (headingText.includes('Factual')) {
                color = percentage > 30 ? '#28a745' : '#6c757d'; // Green if high factual content
            } else if (headingText.includes('Uncertainty')) {
                color = percentage > 15 ? '#ffc107' : '#6c757d'; // Yellow if high uncertainty
            }
            
            circle.querySelectorAll('.progress-circle-bar').forEach(bar => {
                bar.style.borderColor = color;
            });
        } catch (e) {
            console.warn('Error initializing progress circle:', e);
        }
    });
}

/**
 * Initialize charts for data visualization
 */
function initCharts() {
    // Entity type chart
    initEntityTypeChart();
    
    // Domain classification chart
    initDomainChart();
    
    // Explanation chart
    initExplanationChart();
    
    // Model comparison chart
    initModelComparisonChart();
}

function initEntityTypeChart() {
    const chartElement = document.getElementById('entityTypeChart');
    if (!chartElement) return;
    
    const ctx = chartElement.getContext('2d');
    const entityCountsEl = document.getElementById('entityCountsData');
    
    if (!entityCountsEl) return;
    
    try {
        const entityCounts = JSON.parse(entityCountsEl.textContent);
        
        const entityLabels = [];
        const entityValues = [];
        const entityColors = [];
        
        const colorMap = {
            'PERSON': 'rgba(255, 99, 132, 0.8)',
            'ORG': 'rgba(54, 162, 235, 0.8)',
            'LOC': 'rgba(75, 192, 192, 0.8)',
            'GPE': 'rgba(75, 192, 192, 0.8)',
            'DATE': 'rgba(153, 102, 255, 0.8)',
            'TIME': 'rgba(153, 102, 255, 0.8)',
            'MONEY': 'rgba(255, 159, 64, 0.8)',
            'PERCENT': 'rgba(255, 159, 64, 0.8)',
            'PRODUCT': 'rgba(255, 205, 86, 0.8)',
            'EVENT': 'rgba(201, 203, 207, 0.8)',
            'WORK_OF_ART': 'rgba(201, 203, 207, 0.8)',
            'LAW': 'rgba(54, 162, 235, 0.8)',
            'LANGUAGE': 'rgba(255, 99, 132, 0.8)',
            'FAC': 'rgba(75, 192, 192, 0.8)',
            'NORP': 'rgba(153, 102, 255, 0.8)'
        };
        
        for (const [type, count] of Object.entries(entityCounts)) {
            entityLabels.push(type);
            entityValues.push(count);
            entityColors.push(colorMap[type] || 'rgba(128, 128, 128, 0.8)');
        }
        
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: entityLabels,
                datasets: [{
                    data: entityValues,
                    backgroundColor: entityColors,
                    borderColor: entityColors.map(color => color.replace('0.8', '1')),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            boxWidth: 12,
                            font: {
                                size: 11
                            }
                        }
                    }
                }
            }
        });
    } catch (e) {
        console.warn('Error initializing entity chart:', e);
    }
}

function initDomainChart() {
    const chartElement = document.getElementById('domainClassificationChart');
    if (!chartElement) return;
    
    const ctx = chartElement.getContext('2d');
    const domainDataEl = document.getElementById('domainData');
    
    if (!domainDataEl) return;
    
    try {
        const domainData = JSON.parse(domainDataEl.textContent);
        
        const domainLabels = domainData.map(item => item[0].charAt(0).toUpperCase() + item[0].slice(1));
        const domainValues = domainData.map(item => item[1]);
        
        const domainColors = [
            'rgba(255, 99, 132, 0.8)',
            'rgba(54, 162, 235, 0.8)',
            'rgba(75, 192, 192, 0.8)',
            'rgba(153, 102, 255, 0.8)',
            'rgba(255, 159, 64, 0.8)',
            'rgba(255, 205, 86, 0.8)',
            'rgba(201, 203, 207, 0.8)'
        ];
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: domainLabels,
                datasets: [{
                    label: 'Domain Relevance',
                    data: domainValues,
                    backgroundColor: domainColors.slice(0, domainValues.length),
                    borderColor: domainColors.map(color => color.replace('0.8', '1')).slice(0, domainValues.length),
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `References: ${context.raw}`;
                            }
                        }
                    }
                }
            }
        });
    } catch (e) {
        console.warn('Error initializing domain chart:', e);
    }
}

function initExplanationChart() {
    const chartElement = document.getElementById('explanationChart');
    if (!chartElement) return;
    
    const ctx = chartElement.getContext('2d');
    const explanationDataEl = document.getElementById('explanationData');
    
    if (!explanationDataEl) return;
    
    try {
        const explanationItems = JSON.parse(explanationDataEl.textContent);
        const words = explanationItems.map(item => item.word).slice(0, 10); // Take top 10
        const weights = explanationItems.map(item => item.weight).slice(0, 10);
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: words,
                datasets: [{
                    label: 'Feature Impact',
                    data: weights,
                    backgroundColor: weights.map(value => 
                        value >= 0 ? 'rgba(40, 167, 69, 0.7)' : 'rgba(220, 53, 69, 0.7)'
                    ),
                    borderColor: weights.map(value => 
                        value >= 0 ? 'rgba(40, 167, 69, 1)' : 'rgba(220, 53, 69, 1)'
                    ),
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const value = context.raw;
                                return `Impact: ${value.toFixed(4)} (${value >= 0 ? 'Real' : 'Fake'})`;
                            }
                        }
                    }
                }
            }
        });
    } catch (e) {
        console.warn('Error initializing explanation chart:', e);
    }
}

function initModelComparisonChart() {
    const chartElement = document.getElementById('modelComparisonChart');
    if (!chartElement) return;
    
    const ctx = chartElement.getContext('2d');
    const resultsDataEl = document.getElementById('allResultsData');
    
    if (!resultsDataEl) return;
    
    try {
        const allResults = JSON.parse(resultsDataEl.textContent);
        if (!allResults) return;
        
        const models = Object.keys(allResults).filter(m => m !== 'ensemble');
        const confidences = models.map(m => allResults[m].confidence * 100);
        const predictions = models.map(m => allResults[m].prediction);
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: models.map(m => m.charAt(0).toUpperCase() + m.slice(1).replace('_', ' ')),
                datasets: [{
                    label: 'Confidence (%)',
                    data: confidences,
                    backgroundColor: predictions.map(p => 
                        p === 'real' ? 'rgba(40, 167, 69, 0.7)' : 'rgba(220, 53, 69, 0.7)'
                    ),
                    borderColor: predictions.map(p => 
                        p === 'real' ? 'rgba(40, 167, 69, 1)' : 'rgba(220, 53, 69, 1)'
                    ),
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    } catch (e) {
        console.warn('Error initializing model comparison chart:', e);
    }
}
