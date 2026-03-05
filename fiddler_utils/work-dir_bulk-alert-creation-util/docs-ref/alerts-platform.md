# Alerts

### Overview

Fiddler enables users to set up alert rules to track a model's health and performance over time. Fiddler alerts also enable users to dig into triggered alerts and perform root-cause analysis to identify what is causing a model to degrade. Users can set up alerts using the [Fiddler Python client SDK](https://app.gitbook.com/s/rsvU8AIQ2ZL9arerribd/fiddler-python-client-sdk) and the Fiddler UI as demonstrated below.

### Supported Metric Types

You can get alerts for the following metrics:

* [**Traffic**](https://docs.fiddler.ai/observability/platform/traffic-platform)
  * The volume of traffic received by the model over time indicates the overall system's health.
* [**Statistics**](https://docs.fiddler.ai/observability/platform/statistics)
  * Metrics for monitoring basic column aggregations.
* [**Data Drift**](https://docs.fiddler.ai/observability/platform/data-drift-platform)
  * Model performance can be poor if models trained on a specific dataset encounter different data in production.
* [**Data Integrity**](https://docs.fiddler.ai/observability/platform/data-integrity-platform)
  * Three types of violations can occur at model inference: missing feature values, type mismatches (e.g. sending a float input for a categorical feature type) or range mismatches (e.g. sending an unknown US State for a State categorical feature).
* [**Performance**](https://docs.fiddler.ai/observability/platform/performance-tracking-platform)
  * The model performance tells us how well a model performs on its task. A poorly performing model can have significant business implications.
* [**Custom Metrics**](https://docs.fiddler.ai/observability/platform/custom-metrics)
  * Create your own custom metric formulas and create Alerts on those metrics giving you ultimate flexibility in alerting behavior.

### Alert Configurations

#### Comparison Types

There are two options for alert threshold comparison:

* **Absolute** — Compare the metric to an absolute value
  * Example: if traffic for a given hour is less than a threshold of 1,000, trigger alert.
* **Relative** — Compare the metric to a previous period
  * Example: if traffic is down 10% or more than it was at the same time one week ago, trigger alert.

#### Alert Rule Priority & Severity

* **Priority**: Whether you're setting up an alert rule to keep tabs on a model in a test environment or for production scenarios, Fiddler has you covered. Easily set the Alert Rule Priority to indicate the importance of any given Alert Rule. Users can select from Low, Medium, and High priorities.
* **Severity**: Up to two threshold values can be specified for additional flexibility. A **Critical** severity threshold value is always required when setting up an Alert Rule, and a **Warning** threshold value is optional.

### Why do we need alerts?

* It’s not possible to manually track all metrics 24/7.
* Sensible alerts are your first line of defense, and they are meant to warn about issues in production.

#### Setting up Alert Rules

To create a new alert in the Fiddler UI, click Add Alert on the Alerts tab.

1. Fill in the Alert Rule form with basic details like alert name, project, and model.
2. Choose an Alert Type (Traffic, Data Drift, Data Integrity, Performance, Statistic, or Custom Metric) and set up specific metrics, bin size, and columns.
3. Define comparison methods, thresholds, and notification preferences. Click Add Alert Rule to finish.

![](https://3170638587-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F82RHcnYWV62fvrxMeeBB%2Fuploads%2Fgit-blob-9735523ef4001ac9bf4744a9e510481bebbcd8a9%2F52064e5-image.png?alt=media)

In order to create and configure alerts using the Fiddler Python client SDK see [Alert Configuration with Fiddler Client](https://app.gitbook.com/s/jZC6ysdlGhDKECaPCjwm/client-library-reference/alerts-with-fiddler-client).

#### Alert Notification options

You can select the following notification types for your alert.

![](https://3170638587-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F82RHcnYWV62fvrxMeeBB%2Fuploads%2Fgit-blob-5215e83c07796b87b5c95a602600ce7bba488b2a%2Fee80b90-screenshot-2023-10-09-at-51821-pm.png?alt=media)

#### Delete an Alert Rule

Delete an existing alert by clicking on the overflow button (⋮) on the right-hand side of any Alert Rule record and clicking `Delete`. To make any other changes to an Alert Rule, you will need to delete the alert and create a new one with the desired specifications.

![](https://3170638587-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F82RHcnYWV62fvrxMeeBB%2Fuploads%2Fgit-blob-894da3cffb80c502cff03e65d13c55b7f0bc1024%2Feddf05e-image.png?alt=media)

#### Triggered Alert Revisions

Say goodbye to stale alerts! Triggered Alert Revisions mark a leap forward in alert intelligence, giving you the confidence to act decisively and optimize your operations.

Alerts now adapt to changing data. If new information emerges that alters an alert's severity or value, the alert automatically updates you with highlights in the user interface and revised notifications. This approach empowers you to:

* Make informed decisions based on real-time data: No more relying on outdated or inaccurate alerts.
* Focus on critical issues: Updated alerts prioritize the most relevant information.

![Inspect Alert experience](https://3170638587-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F82RHcnYWV62fvrxMeeBB%2Fuploads%2Fgit-blob-85090507fc098375a69345495ce60da475bbd18c%2F5921286-screenshot-2024-03-07-at-52104-pm.png?alt=media)

Inspect Alert experience

![Triggered Alert revision experience](https://3170638587-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F82RHcnYWV62fvrxMeeBB%2Fuploads%2Fgit-blob-69395b1fae22a72b2b34762fb150861777ff4c74%2F7f0aa27-screenshot-2024-03-07-at-52113-pm.png?alt=media)

### Sample Alert Email

Here's a sample of an email that's sent if an alert is triggered:

![](https://3170638587-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F82RHcnYWV62fvrxMeeBB%2Fuploads%2Fgit-blob-e8af788c855b17078deb431c4bfea215582b3ec1%2Falert-email-perf-example.png?alt=media)

### Integrations

The Integrations tab is a read-only view of all the integrations your Admin has enabled for use. As of today, users can configure their Alert Rules to notify them via email or Pager Duty services.

![](https://3170638587-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F82RHcnYWV62fvrxMeeBB%2Fuploads%2Fgit-blob-9df9e3880d2491f8fe7e74e6599b8d993004d7b6%2F7462149-image.png?alt=media)

Admins can add new integrations by clicking on the setting cog icon in the main navigation bar and selecting the integration tab of interest.

![](https://3170638587-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F82RHcnYWV62fvrxMeeBB%2Fuploads%2Fgit-blob-9e977eec34bebac743078f53174b3dd85984ee68%2F6ee3027-screen-shot-2022-10-03-at-41600-pm.png?alt=media)

#### Pause alert notification

This feature allows users to temporarily pause and resume notifications for specific alerts without affecting their evaluation and triggering mechanisms. It enhances user experience by providing efficient notification management.\\

**How to Use**

**Using the Fiddler User Interface (UI)**

* Locate the Alert Tool:\
  Navigate to the alert rule table and identify the desired alert.
* Toggle Notifications:
  * Click the notification bell icon.
  * The icon updates to indicate the new state (paused or resumed).
* Confirm Action:
  * A loading indicator and a toast notification confirm the action.

**Using the Fiddler Client SDK**

For programmatic control, use the Fiddler client SDK's alert-rules method with the enable\_notification argument.

* Details:\
  Refer to the [Fiddler Python client SDK Reference](https://app.gitbook.com/s/rsvU8AIQ2ZL9arerribd/fiddler-langgraph-sdk/core/fiddler-client) for a complete explanation of SDK features.

**Note**

* No Impact on Evaluation:\
  Pausing notifications does not affect the evaluation of alert conditions. The alert tool will continue to assess conditions and trigger alerts as usual.

***

:question: Questions? [Talk](https://www.fiddler.ai/contact-sales) to a product expert or [request](https://www.fiddler.ai/demo) a demo.

:bulb: Need help? Contact us at <help@fiddler.ai>.
