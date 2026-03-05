# AlertRecord

Alert record representing a triggered alert instance.

An AlertRecord captures the details of a specific alert trigger event, including the metric values, thresholds, and context that caused an AlertRule to fire. Alert records provide essential data for monitoring analysis and troubleshooting.

## Example

```python
# List recent critical alerts
critical_alerts = [
    record for record in AlertRecord.list(
        alert_rule_id=drift_alert.id,
        start_time=datetime.now() - timedelta(days=3)
    )
    if record.severity == "CRITICAL"
]

# Analyze alert details
for alert in critical_alerts:

    print(f"Alert triggered at {alert.created_at}")
    print(f"Metric value: {alert.alert_value:.3f}")
    print(f"Critical threshold: {alert.critical_threshold:.3f}")
    if alert.feature_name:

        print(f"Feature: {alert.feature_name}")

        print(f"Message: {alert.message}")
        print("---")

        # Check for alert patterns
        hourly_alerts = {}
        for alert in AlertRecord.list(alert_rule_id=perf_alert.id):

            hour = alert.created_at.hour
            hourly_alerts[hour] = hourly_alerts.get(hour, 0) + 1

            print("Alerts by hour:", hourly_alerts)
```

{% hint style="info" %}
Alert records are read-only entities created automatically by the Fiddler platform when AlertRules trigger. They cannot be created or modified directly but provide valuable historical data for analysis and debugging.
{% endhint %}

Initialize an AlertRecord instance.

Creates an alert record object for representing triggered alert instances. Alert records are typically created automatically by the Fiddler platform when AlertRules trigger, rather than being instantiated directly by users.

{% hint style="info" %}
Alert records are read-only entities that capture historical alert trigger events. They are created automatically by the system and cannot be modified after creation.

**Return type:** None
{% endhint %}

## *classmethod* list(alert\_rule\_id, start\_time=None, end\_time=None, ordering=None)

List alert records triggered by a specific alert rule.

Retrieves historical alert records for analysis and troubleshooting. This method provides access to all alert trigger events within a specified time range, enabling pattern analysis and threshold tuning.

## Parameters

| Parameter       | Type                | Required | Default | Description                                                                                                               |
| --------------- | ------------------- | -------- | ------- | ------------------------------------------------------------------------------------------------------------------------- |
| `alert_rule_id` | \`UUID              | str\`    | ✗       | `None`                                                                                                                    |
| `start_time`    | \`datetime          | None\`   | ✗       | `7 days ago`                                                                                                              |
| `end_time`      | \`datetime          | None\`   | ✗       | `current time`                                                                                                            |
| `ordering`      | `list[str] \| None` | ✗        | `None`  | List of field names for result ordering. Prefix with "-" for descending order (e.g., \["-created\_at"] for newest first). |

## Yields

`AlertRecord` -- Alert record instances with complete trigger details and context information. **Return type:** *Iterator*\[[*AlertRecord*](#fiddler.entities.alert_record.AlertRecord)]

## Example

```python
# Get recent alerts for analysis
recent_alerts = list(AlertRecord.list(
    alert_rule_id=drift_alert.id,
    start_time=datetime.now() - timedelta(days=3),
    ordering=["-created_at"]  # Newest first
))

# Analyze alert frequency
print(f"Total alerts in last 3 days: {len(recent_alerts)}")
critical_count = sum(1 for a in recent_alerts if a.severity == "CRITICAL")
print(f"Critical alerts: {critical_count}")

# Check alert patterns by feature
feature_alerts = {}
for alert in recent_alerts:

    if alert.feature_name:
        feature_alerts[alert.feature_name] = feature_alerts.get(alert.feature_name, 0) + 1

        print("Alerts by feature:", feature_alerts)

        # Analyze threshold violations
        for alert in recent_alerts[:5]:  # Latest 5 alerts

        violation_ratio = alert.alert_value / alert.critical_threshold
        print(f"Alert value: {alert.alert_value:.3f} "

        f"({violation_ratio:.1%} of threshold)")
```

{% hint style="info" %}
Results are paginated automatically. The default time range is 7 days to balance performance with useful historical context. Use ordering parameters to get the most relevant results first.
{% endhint %}
