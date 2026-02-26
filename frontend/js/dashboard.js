/**
 * Dashboard chart rendering using Chart.js.
 */

let timelineChart = null;
let emotionChart = null;

async function loadDashboard() {
    try {
        // Load summary stats
        const summary = await API.request('/api/analytics/summary');
        document.getElementById('stat-total').textContent = summary.total_conversations;
        document.getElementById('stat-active').textContent = summary.active_conversations;
        document.getElementById('stat-escalated').textContent = summary.escalated_conversations;
        document.getElementById('stat-messages').textContent = summary.total_messages;
        document.getElementById('stat-urgent').textContent = summary.urgent_messages;
        document.getElementById('stat-today').textContent = summary.conversations_today;

        // Sentiment display
        const sentVal = document.getElementById('sentiment-value');
        const sentLabel = document.getElementById('sentiment-label');
        const s = summary.avg_sentiment;
        sentVal.textContent = s.toFixed(2);
        if (s > 0.2) {
            sentVal.style.color = '#10b981';
            sentLabel.textContent = '😊 Overall Positive';
        } else if (s < -0.2) {
            sentVal.style.color = '#ef4444';
            sentLabel.textContent = '😟 Overall Negative';
        } else {
            sentVal.style.color = '#3b82f6';
            sentLabel.textContent = '😐 Neutral';
        }

        // Emotion pie chart
        renderEmotionChart(summary.top_emotions);

        // Timeline
        const timeline = await API.request('/api/analytics/timeline?days=7');
        renderTimelineChart(timeline.timeline);

    } catch (e) {
        console.error('Dashboard load error:', e);
        Toast.error('Failed to load dashboard data');
    }
}

function renderTimelineChart(data) {
    const ctx = document.getElementById('timeline-chart');
    if (!ctx) return;

    if (timelineChart) timelineChart.destroy();

    timelineChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(d => d.date),
            datasets: [{
                label: 'Conversations',
                data: data.map(d => d.conversations),
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#6366f1',
                pointBorderColor: '#6366f1',
                pointRadius: 4,
                pointHoverRadius: 6,
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false,
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#64748b', font: { family: 'Inter' } },
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#64748b', font: { family: 'Inter' }, stepSize: 1 },
                    beginAtZero: true,
                }
            }
        }
    });
}

function renderEmotionChart(emotions) {
    const ctx = document.getElementById('emotion-chart');
    if (!ctx) return;

    if (emotionChart) emotionChart.destroy();

    const labels = Object.keys(emotions);
    const values = Object.values(emotions);

    if (labels.length === 0) {
        labels.push('No data');
        values.push(1);
    }

    const colors = {
        very_positive: '#10b981',
        positive: '#34d399',
        neutral: '#3b82f6',
        negative: '#f59e0b',
        very_negative: '#ef4444',
        'No data': '#334155',
    };

    emotionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: labels.map(l => colors[l] || '#6366f1'),
                borderColor: 'rgba(0,0,0,0.3)',
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#94a3b8',
                        font: { family: 'Inter', size: 12 },
                        padding: 16,
                    }
                }
            },
            cutout: '60%',
        }
    });
}
