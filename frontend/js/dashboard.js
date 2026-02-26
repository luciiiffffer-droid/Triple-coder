/**
 * Dashboard charts — Plus Jakarta Sans font, new color palette.
 */

let timelineChart = null;
let emotionChart = null;

async function loadDashboard() {
    try {
        const summary = await API.request('/api/analytics/summary');
        document.getElementById('stat-total').textContent = summary.total_conversations;
        document.getElementById('stat-active').textContent = summary.active_conversations;
        document.getElementById('stat-escalated').textContent = summary.escalated_conversations;
        document.getElementById('stat-messages').textContent = summary.total_messages;
        document.getElementById('stat-urgent').textContent = summary.urgent_messages;
        document.getElementById('stat-today').textContent = summary.conversations_today;

        const sentVal = document.getElementById('sentiment-value');
        const sentLabel = document.getElementById('sentiment-label');
        const s = summary.avg_sentiment;
        sentVal.textContent = s.toFixed(2);
        if (s > 0.2) {
            sentVal.style.color = '#10b981';
            sentLabel.textContent = '😊 Overall Positive';
        } else if (s < -0.2) {
            sentVal.style.color = '#f43f5e';
            sentLabel.textContent = '😟 Overall Negative';
        } else {
            sentVal.style.color = '#06b6d4';
            sentLabel.textContent = '😐 Neutral';
        }

        renderEmotionChart(summary.top_emotions);

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
                backgroundColor: 'rgba(99,102,241,0.08)',
                fill: true,
                tension: 0.45,
                pointBackgroundColor: '#8b5cf6',
                pointBorderColor: '#6366f1',
                pointBorderWidth: 2,
                pointRadius: 5,
                pointHoverRadius: 7,
                borderWidth: 2.5,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { color: '#4a5568', font: { family: 'Plus Jakarta Sans', size: 11 } },
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { color: '#4a5568', font: { family: 'Plus Jakarta Sans', size: 11 }, stepSize: 1 },
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

    if (labels.length === 0) { labels.push('No data'); values.push(1); }

    const colors = {
        very_positive: '#10b981',
        positive: '#34d399',
        neutral: '#06b6d4',
        negative: '#f59e0b',
        very_negative: '#f43f5e',
        'No data': '#1e2a40',
    };

    emotionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: labels.map(l => (colors[l] || '#6366f1') + 'cc'),
                borderColor: labels.map(l => colors[l] || '#6366f1'),
                borderWidth: 2,
                hoverOffset: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#8892b0',
                        font: { family: 'Plus Jakarta Sans', size: 12 },
                        padding: 16,
                        usePointStyle: true,
                        pointStyleWidth: 8,
                    }
                }
            },
            cutout: '62%',
        }
    });
}
