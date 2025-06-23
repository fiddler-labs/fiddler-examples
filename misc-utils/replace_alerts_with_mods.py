import fiddler as fdl
from tqdm import tqdm
import logging
from datetime import datetime
import os

# Set up logging
def setup_logging(log_level: str = "INFO"):
    """
    Set up logging with appropriate configuration
    Args: log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Clear any existing handlers from the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure the root logger
    log_format = '%(asctime)s - %(levelname)8s - %(lineno)4d - %(name)s - %(module)s - %(funcName)s - %(exc_info)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    app_log_file = f"logs/log_{timestamp}.log"
    fiddler_log_file = f"logs/fiddler_log_{timestamp}.log"
    
    # Create a filter for Fiddler library logs (but not our application's fiddler-related module)
    class FiddlerLibraryLogFilter(logging.Filter):
        def filter(self, record):
            return record.name.startswith(('fiddler.', 'urllib3.'))
    
    # Create the inverse filter for application logs (excluding Fiddler library logs)
    class AppLogFilter(logging.Filter):
        def filter(self, record):
            return not record.name.startswith(('fiddler.', 'urllib3.'))
    
    # Add file handler for application logs
    app_file_handler = logging.FileHandler(app_log_file)
    app_file_handler.setFormatter(formatter)
    app_file_handler.setLevel(numeric_level)
    app_file_handler.addFilter(AppLogFilter())
    
    # Add file handler for Fiddler logs
    fiddler_file_handler = logging.FileHandler(fiddler_log_file)
    fiddler_file_handler.setFormatter(formatter)
    fiddler_file_handler.setLevel(logging.INFO)
    fiddler_file_handler.addFilter(FiddlerLibraryLogFilter())
    
    # Add console handler only for ERROR and above (for stack traces)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.ERROR)
    
    # Add handlers to the root logger
    root_logger.addHandler(app_file_handler)
    root_logger.addHandler(fiddler_file_handler)
    root_logger.addHandler(console_handler)
    
    # Set the root logger level
    root_logger.setLevel(numeric_level)
    
    print(f"Application logs will be written to: {app_log_file}")
    print(f"Fiddler library logs will be written to: {fiddler_log_file}")

# Setup logging with default values
setup_logging(log_level="DEBUG")
logger = logging.getLogger(__name__)


# Authentication credentials
URL = 'https://customer.fiddler.ai'
TOKEN = 'ADD_YOUR_TOKEN_HERE'

# Initialize Fiddler client
fdl.init(url=URL, token=TOKEN)
logger.info(f"Successfully connected to Fiddler at {URL}")
logger.info(f"Client version: {fdl.__version__}")
logger.info(f"Server version:    {fdl.conn.server_version}")
logger.info(f"Organization ID:   {fdl.conn.organization_id}")
logger.info(f"Organization name: {fdl.conn.organization_name}")

# Get all projects
projects = list(fdl.Project.list())
logger.info(f"Found {len(projects)} projects")

# Initialize counters
total_alerts = 0
monthly_alerts = 0
successful_changes = 0


for project in tqdm(projects, desc="Processing projects" , leave=True):
    project_name = project.name
    logger.info(f"Processing project: {project_name}")
    
    if project.id is not None :
        models = list(fdl.Model.list(project_id=project.id))
        logger.info(f"  Found {len(models)} models in project {project_name}")
        
        for model in tqdm(models, desc=f"    Processing models for project {project_name}" , leave=True):
            model_name = model.name
            logger.debug(f"    Processing model: {model_name}")
            
            try:
                model_alerts = list(fdl.AlertRule.list(model_id=model.id))
                total_alerts += len(model_alerts)
                logger.debug(f"      Found {len(model_alerts)} alerts for model {model_name}")
                
                # Process each alert
                for alert in tqdm(model_alerts, desc=f"        Processing alerts for model {model_name}" , leave=True):
                    # Check if this is a monthly alert
                    if alert.bin_size == fdl.BinSize.MONTH:
                        monthly_alerts += 1
                        logger.debug(f"      Processing monthly alert: {alert.name}")
                        
                        # Create a duplicate alert with daily bin size
                        try:
                            # Get all properties from existing alert
                            new_alert = fdl.AlertRule(
                                name=f"{alert.name}_FIXED",
                                bin_size=fdl.BinSize.DAY,  # Change to daily bin size
                                
                                model_id=alert.model_id,
                                metric_id=alert.metric_id,
                                priority=alert.priority,
                                compare_to=alert.compare_to,
                                condition=alert.condition,
                                segment_id=alert.segment_id,
                                threshold_type=alert.threshold_type,
                                critical_threshold=alert.critical_threshold,
                                warning_threshold=alert.warning_threshold,
                                columns=alert.columns,
                                compare_bin_delta=alert.compare_bin_delta,
                                evaluation_delay=alert.evaluation_delay,
                                category=alert.category,
                                baseline_id=alert.baseline_id,
                                )
                            
                            new_alert.create()
                            logger.debug(f"      Created new daily alert: {new_alert.name}")
                            
                            alert.delete()
                            logger.debug(f"      Deleted original monthly alert: {alert.name}")
                            
                            successful_changes += 1
                        except Exception as e:
                            logger.debug(f"      Error processing alert {alert.name}: {str(e)}")
                            
            except Exception as e:
                logger.error(f"      Error retrieving alerts for model {model_name}: {str(e)}")
                continue
    else:
        logger.error(f"  Skipping project {project_name} - no project ID available")

logger.info("\nSUMMARY:")
logger.info(f"Total alerts found: {total_alerts}")
logger.info(f"Monthly alerts found: {monthly_alerts}")
logger.info(f"Successfully converted: {successful_changes}")


