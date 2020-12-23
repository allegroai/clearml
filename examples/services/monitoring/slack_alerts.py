"""
Create a ClearML Monitoring Service that posts alerts on Slack Channel groups based on some logic

Creating a new Slack Bot (Allegro ClearML Bot):
1. Login to your Slack account
2. Go to https://api.slack.com/apps/new
3. Give the new App a name (For example "Allegro Train Bot") and select your workspace
4. Press Create App
5. In "Basic Information" under "Display Information" fill in the following fields
    - In "Short description" insert "Allegro Train Bot"
    - In "Background color" insert #202432
6. Press Save Changes
7. In "OAuth & Permissions" under "Scopes" click on "Add an OAuth Scope" and
   select from the drop down list the following three permissions:
        channels:join
        channels:read
        chat:write
8. Now under "OAuth Tokens & Redirect URLs" press on "Install App to Workspace",
   then hit "Allow" on the confirmation dialog
9. Under "OAuth Tokens & Redirect URLs" copy the "Bot User OAuth Access Token" by clicking on "Copy" button
10. To use the copied API Token in the Allegro ClearML Slack service,
    execute the script with --slack_api "<api_token_here>"  (notice the use of double quotes around the token)

We are done!
"""

import argparse
import os
from time import sleep
from typing import Optional

from slack import WebClient
from slack.errors import SlackApiError

from clearml import Task
from clearml.automation.monitor import Monitor


class SlackMonitor(Monitor):
    """
    Create a monitoring service that alerts on Task failures / completion in a Slack channel
    """
    def __init__(self, slack_api_token, channel, message_prefix=None):
        # type: (str, str, Optional[str]) -> ()
        """
        Create a Slack Monitoring object.
        It will alert on any Task/Experiment that failed or completed

        :param slack_api_token: Slack bot API Token. Token should start with "xoxb-"
        :param channel: Name of the channel to post alerts to
        :param message_prefix: optional message prefix to add before any message posted
            For example: message_prefix="Hey <!here>,"
        """
        super(SlackMonitor, self).__init__()
        self.channel = '{}'.format(channel[1:] if channel[0] == '#' else channel)
        self.slack_client = WebClient(token=slack_api_token)
        self.min_num_iterations = 0
        self.status_alerts = ["failed", ]
        self.include_manual_experiments = False
        self._channel_id = None
        self._message_prefix = '{} '.format(message_prefix) if message_prefix else ''
        self.check_credentials()

    def check_credentials(self):
        # type: () -> ()
        """
        Check we have the correct credentials for the slack channel
        """
        self.slack_client.api_test()

        # Find channel ID
        response = self.slack_client.conversations_list()
        channel_id = [channel_info.get('id') for channel_info in response.data['channels']
                      if channel_info.get('name') == self.channel]
        if not channel_id:
            raise ValueError('Error: Could not locate channel name \'{}\''.format(self.channel))

        # test bot permission (join channel)
        self._channel_id = channel_id[0]
        self.slack_client.conversations_join(channel=self._channel_id)

    def post_message(self, message, retries=1, wait_period=10.):
        # type: (str, int, float) -> ()
        """
        Post message on our slack channel

        :param message: Message to be sent (markdown style)
        :param retries: Number of retries before giving up
        :param wait_period: wait between retries in seconds
        """
        for i in range(retries):
            if i != 0:
                sleep(wait_period)

            try:
                self.slack_client.chat_postMessage(
                    channel=self._channel_id,
                    blocks=[dict(type="section", text={"type": "mrkdwn", "text": message})],
                )
                return
            except SlackApiError as e:
                print("While trying to send message: \"\n{}\n\"\nGot an error: {}".format(
                    message, e.response['error']))

    def get_query_parameters(self):
        # type: () -> dict
        """
        Return the query parameters for the monitoring.

        :return dict: Example dictionary: {'status': ['failed'], 'order_by': ['-last_update']}
        """
        filter_tags = ['-archived'] + (['-development'] if not self.include_manual_experiments else [])
        return dict(status=self.status_alerts, order_by=['-last_update'], system_tags=filter_tags)

    def process_task(self, task):
        """
        # type: (Task) -> ()
        Called on every Task that we monitor.
        This is where we send the Slack alert

        :return: None
        """
        # skipping failed tasks with low number of iterations
        if self.min_num_iterations and task.get_last_iteration() < self.min_num_iterations:
            print('Skipping {} experiment id={}, number of iterations {} < {}'.format(
                task.status, task.id, task.get_last_iteration(), self.min_num_iterations))
            return

        print('Experiment id={} {}, raising alert on channel \"{}\"'.format(task.id, task.status, self.channel))

        console_output = task.get_reported_console_output(number_of_reports=3)
        message = \
            '{}Experiment ID <{}|{}> *{}*\nProject: *{}*  -  Name: *{}*\n' \
            '```\n{}\n```'.format(
                self._message_prefix,
                task.get_output_log_web_page(), task.id,
                task.status,
                task.get_project_name(), task.name,
                ('\n'.join(console_output))[-2048:])
        self.post_message(message, retries=5)


def main():
    print('ClearML experiment monitor Slack service\n')

    # Slack Monitor arguments
    parser = argparse.ArgumentParser(description='ClearML monitor experiments and post Slack Alerts')
    parser.add_argument('--channel', type=str,
                        help='Set the channel to post the Slack alerts')
    parser.add_argument('--slack_api', type=str, default=os.environ.get('SLACK_API_TOKEN', None),
                        help='Slack API key for sending messages')
    parser.add_argument('--message_prefix', type=str,
                        help='Add message prefix (For example, to alert all channel members use: "Hey <!here>,")')
    parser.add_argument('--project', type=str, default='',
                        help='The name (or partial name) of the project to monitor, use empty for all projects')
    parser.add_argument('--min_num_iterations', type=int, default=0,
                        help='Minimum number of iterations of failed/completed experiment to alert. '
                             'This will help eliminate unnecessary debug sessions that crashed right after starting '
                             '(default:0 alert on all)')
    parser.add_argument('--include_manual_experiments', action="store_true", default=False,
                        help='Include experiments running manually (i.e. not by clearml-agent)')
    parser.add_argument('--include_completed_experiments', action="store_true", default=False,
                        help='Include completed experiments (i.e. not just failed experiments)')
    parser.add_argument('--refresh_rate', type=float, default=10.,
                        help='Set refresh rate of the monitoring service, default every 10.0 sec')
    parser.add_argument('--service_queue', type=str, default='services',
                        help='Queue name to use when running as a service (default: \'services\'')
    parser.add_argument('--local', action="store_true", default=False,
                        help='Run service locally instead of as a service '
                             '(Default: Automatically launch itself on the services queue)')

    args = parser.parse_args()

    if not args.slack_api:
        print('Slack API key was not provided, please run with --slack_api <KEY>')
        exit(1)

    if not args.channel:
        print('Slack channel was not provided, please run with --channel <channel_name>')
        exit(1)

    # create the slack monitoring object
    slack_monitor = SlackMonitor(
        slack_api_token=args.slack_api, channel=args.channel, message_prefix=args.message_prefix)

    # configure the monitoring filters
    slack_monitor.min_num_iterations = args.min_num_iterations
    slack_monitor.include_manual_experiments = args.include_manual_experiments
    if args.project:
        slack_monitor.set_projects(project_names_re=[args.project])
    if args.include_completed_experiments:
        slack_monitor.status_alerts += ["completed"]

    # start the monitoring Task
    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    task = Task.init(project_name='Monitoring', task_name='Slack Alerts', task_type=Task.TaskTypes.monitor)
    if not args.local:
        task.execute_remotely(queue_name=args.service_queue)
        # we will not get here if we are running locally

    print('\nStarting monitoring service\nProject: "{}"\nRefresh rate: {}s\n'.format(
        args.project or 'all', args.refresh_rate))

    # Let everyone know we are up and running
    start_message = \
        '{}Allegro ClearML Slack monitoring service started\nMonitoring project \'{}\''.format(
            (args.message_prefix + ' ') if args.message_prefix else '',
            args.project or 'all')
    slack_monitor.post_message(start_message)

    # Start the monitor service, this function will never end
    slack_monitor.monitor(pool_period=args.refresh_rate)


if __name__ == "__main__":
    main()
