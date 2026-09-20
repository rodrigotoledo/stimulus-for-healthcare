import { Application } from "@hotwired/stimulus";
import DatasetBrowserController from "controllers/dataset_browser_controller";
import DialogExplorerController from "controllers/dialog_explorer_controller";
import AnalysisStreamController from "controllers/analysis_stream_controller";
import StatusController from "controllers/status_controller";

const application = Application.start();
application.debug = false;
window.Stimulus = application;

application.register("dataset-browser", DatasetBrowserController);
application.register("dialog-explorer", DialogExplorerController);
application.register("analysis-stream", AnalysisStreamController);
application.register("status", StatusController);
