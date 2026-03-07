module.exports = {
  apps: [
    {
      name: 'tweet-predictor',
      script: '/root/claude-chat/tweet_predict_skill.sh',
      args: '--live --update',
      cron_restart: '0 */1 * * *',  // 每小时执行一次
      autorestart: false,            // 脚本跑完不自动重启
      watch: false,
      log_file: '/root/claude-chat/logs/tweet_predict.log',
      error_file: '/root/claude-chat/logs/tweet_predict_error.log',
      out_file: '/root/claude-chat/logs/tweet_predict_out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      env: {
        HOME: '/root'
      }
    }
  ]
};
