# AlertRule

Alert rule for automated monitoring and alerting in ML systems.

An AlertRule defines conditions that automatically trigger notifications when ML model metrics exceed specified thresholds. Alert rules are essential for proactive monitoring of model performance, data drift, and operational issues.

## Example

```python
# Create feature drift alert
drift_alert = AlertRule(
    name="credit_score_drift",
    model_id=model.id,
    metric_id="drift_score",
    priority=Priority.HIGH,
    compare_to=CompareTo.BASELINE,
    condition=AlertCondition.GT,
    bin_size=BinSize.HOUR,
    critical_threshold=0.8,
    warning_threshold=0.6,
    baseline_id=baseline.id,
    columns=["credit_score", "income"]
).create()

# Create performance degradation alert
perf_alert = AlertRule(
    name="accuracy_drop",
    model_id=model.id,
    metric_id="accuracy",
    priority=Priority.MEDIUM,
    compare_to=CompareTo.TIME_PERIOD,
    condition=AlertCondition.LESSER,
    bin_size=BinSize.DAY,
    critical_threshold=0.85,
    compare_bin_delta=7  # Compare to 7 days ago
).create()

# Configure notifications
drift_alert.set_notification_config(
    emails=["[ml-team@company.com](mailto:ml-team@company.com)", "[data-team@company.com](mailto:data-team@company.com)"],
    pagerduty_services=["ML_ALERTS"],
    pagerduty_severity="critical"
)
```

{% hint style="info" %}
Alert rules continuously monitor metrics and trigger notifications when thresholds are exceeded. Use appropriate evaluation delays to avoid false positives from temporary data fluctuations.
{% endhint %}

Initialize an AlertRule instance.

Creates an alert rule configuration for automated monitoring of ML model metrics. The alert rule defines conditions that trigger notifications when thresholds are exceeded, enabling proactive monitoring of model performance and data quality.

## Parameters

| Parameter               | Type                     | Required | Default | Description                                                                                            |
| ----------------------- | ------------------------ | -------- | ------- | ------------------------------------------------------------------------------------------------------ |
| `name`                  | `str`                    | ✗        | `None`  | Human-readable name for the alert rule. Should be descriptive and unique within the model context.     |
| `model_id`              | \`UUID                   | str\`    | ✗       | `None`                                                                                                 |
| `metric_id`             | \`str                    | UUID\`   | ✗       | `None`                                                                                                 |
| `priority`              | \`Priority               | str\`    | ✗       | `None`                                                                                                 |
| `compare_to`            | \`CompareTo              | str\`    | ✗       | `None`                                                                                                 |
| `condition`             | \`AlertCondition         | str\`    | ✗       | `None`                                                                                                 |
| `bin_size`              | \`BinSize                | str\`    | ✗       | `None`                                                                                                 |
| `threshold_type`        | \`AlertThresholdAlgo     | str\`    | ✗       | `None`                                                                                                 |
| `auto_threshold_params` | `dict[str, Any] \| None` | ✗        | `None`  | Parameters for automatic threshold calculation. Used when threshold\_type is `AUTO`.                   |
| `critical_threshold`    | \`float                  | None\`   | ✗       | `None`                                                                                                 |
| `warning_threshold`     | \`float                  | None\`   | ✗       | `None`                                                                                                 |
| `columns`               | `list[str] \| None`      | ✗        | `None`  | List of feature columns to monitor. For feature-specific drift alerts. If None, monitors all features. |
| `baseline_id`           | \`UUID                   | str      | None\`  | ✗                                                                                                      |
| `segment_id`            | \`UUID                   | str      | None\`  | ✗                                                                                                      |
| `compare_bin_delta`     | \`int                    | None\`   | ✗       | `None`                                                                                                 |
| `evaluation_delay`      | `int`                    | ✗        | `None`  | Delay in minutes before evaluating alerts. Helps avoid false positives from incomplete data.           |
| `category`              | \`str                    | None\`   | ✗       | `None`                                                                                                 |

## Example

```python
# Feature drift alert with baseline comparison
drift_alert = AlertRule(
    name="income_drift_detection",
    model_id=model.id,
    metric_id="drift_score",
    priority=Priority.HIGH,
    compare_to=CompareTo.BASELINE,
    condition=AlertCondition.GT,
    bin_size=BinSize.HOUR,
    critical_threshold=0.8,
    warning_threshold=0.6,
    baseline_id=baseline.id,
    columns=["income", "credit_score"],
    evaluation_delay=15,  # 15 minute delay
    category="data_quality"
)

# Performance monitoring with time comparison
perf_alert = AlertRule(
    name="weekly_accuracy_check",
    model_id=model.id,
    metric_id="accuracy",
    priority=Priority.MEDIUM,
    compare_to=CompareTo.TIME_PERIOD,
    condition=AlertCondition.LESSER,
    bin_size=BinSize.DAY,
    critical_threshold=0.85,
    compare_bin_delta=7,  # Compare to 7 days ago
    category="performance"
)
```

{% hint style="info" %}
After initialization, call create() to persist the alert rule to the Fiddler platform. Alert rules begin monitoring immediately after creation.
{% endhint %}

## *classmethod* get(id\_)

Retrieve an alert rule by its unique identifier.

Fetches an alert rule from the Fiddler platform using its UUID. This method returns the complete alert rule configuration including thresholds, notification settings, and monitoring status.

## Parameters

| Parameter | Type   | Required | Default | Description |
| --------- | ------ | -------- | ------- | ----------- |
| `id_`     | \`UUID | str\`    | ✗       | `None`      |

## Returns

The alert rule instance with all configuration and metadata populated from the server.

**Return type:** `AlertRule`

## Raises

* **NotFound** -- If no alert rule exists with the specified ID.
* **ApiError** -- If there's an error communicating with the Fiddler API.

## Example

```python
# Retrieve alert rule by ID
alert_rule = AlertRule.get(id_="550e8400-e29b-41d4-a716-446655440000")
print(f"Alert: {alert_rule.name}")
print(f"Metric: {alert_rule.metric_id}")
print(f"Priority: {alert_rule.priority}")
print(f"Critical threshold: {alert_rule.critical_threshold}")

# Check notification configuration
notification_config = alert_rule.get_notification_config()
print(f"Email recipients: {notification_config.emails}")
```

{% hint style="info" %}
This method makes an API call to fetch the latest alert rule configuration from the server, including any recent threshold or notification updates.
{% endhint %}

## *classmethod* list(model\_id, metric\_id=None, columns=None, baseline\_id=None, ordering=None)

Get a list of all alert rules in the organization.

## Parameters

| Parameter     | Type                | Required | Default | Description                                                          |
| ------------- | ------------------- | -------- | ------- | -------------------------------------------------------------------- |
| `model_id`    | \`UUID              | str\`    | ✗       | `None`                                                               |
| `metric_id`   | \`UUID              | str      | None\`  | ✗                                                                    |
| `columns`     | `list[str] \| None` | ✗        | `None`  | list rules set on the specified list of columns                      |
| `baseline_id` | \`UUID              | str      | None\`  | ✗                                                                    |
| `ordering`    | `list[str] \| None` | ✗        | `None`  | order result as per list of fields. \["-field\_name"] for descending |

## Returns

paginated list of alert rules for the specified filters

**Return type:** *Iterator*\[[*AlertRule*](#fiddler.entities.alert_rule.AlertRule)]

## delete()

Delete an alert rule.

**Return type:** None

## create()

Create a new alert rule.

**Return type:** [*AlertRule*](#fiddler.entities.alert_rule.AlertRule)

## update()

Update an existing alert rule.

**Return type:** None

## enable\_notifications()

Enable notifications for an alert rule

**Return type:** None

## disable\_notifications()

Disable notifications for an alert rule

**Return type:** None

## set\_notification\_config()

Set notification config for an alert rule

## Parameters

| Parameter            | Type                 | Required | Default | Description                |
| -------------------- | -------------------- | -------- | ------- | -------------------------- |
| `emails`             | `list[str] \| None`  | ✗        | `None`  | list of emails             |
| `pagerduty_services` | `list[str] \| None`  | ✗        | `None`  | list of pagerduty services |
| `pagerduty_severity` | \`str                | None\`   | ✗       | `None`                     |
| `webhooks`           | `list[UUID] \| None` | ✗        | `None`  | list of webhooks UUIDs     |

## Returns

NotificationConfig object

**Return type:** *NotificationConfig*

## get\_notification\_config()

Get notifications config for an alert rule

## Returns

NotificationConfig object

**Return type:** *NotificationConfig*
