import React from 'react';
import { FileText, CheckCircle2, Clock, ThumbsUp, ArrowUpRight } from 'lucide-react';

export default function SystemHealthCards({ kpis, documents }) {
  const activeCount = documents.filter(d => d.status === 'Active').length;
  const totalCount = documents.length;

  const cards = [
    {
      title: "Total Knowledge Documents",
      value: totalCount,
      change: "+2 this month",
      icon: FileText,
      color: "indigo",
      bgGradient: "from-indigo-500/10 to-indigo-600/5",
      borderColor: "border-indigo-200 dark:border-indigo-900/60",
      textColor: "text-indigo-600 dark:text-indigo-400"
    },
    {
      title: "Active Knowledge Sources",
      value: `${activeCount} / ${totalCount}`,
      change: "Indexed in CDSS consensus",
      icon: CheckCircle2,
      color: "emerald",
      bgGradient: "from-emerald-500/10 to-emerald-600/5",
      borderColor: "border-emerald-200 dark:border-emerald-900/60",
      textColor: "text-emerald-600 dark:text-emerald-400"
    },
    {
      title: "Average Query Latency",
      value: `${kpis.avgQueryResponseTimeMs} ms`,
      change: "12% faster than target",
      icon: Clock,
      color: "cyan",
      bgGradient: "from-cyan-500/10 to-cyan-600/5",
      borderColor: "border-cyan-200 dark:border-cyan-900/60",
      textColor: "text-cyan-600 dark:text-cyan-400"
    },
    {
      title: "Clinician Satisfaction",
      value: `${kpis.userSatisfactionScore}%`,
      change: "Based on 340+ verified citations",
      icon: ThumbsUp,
      color: "amber",
      bgGradient: "from-amber-500/10 to-amber-600/5",
      borderColor: "border-amber-200 dark:border-amber-900/60",
      textColor: "text-amber-600 dark:text-amber-400"
    }
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      {cards.map((card, idx) => {
        const IconComponent = card.icon;
        return (
          <div
            key={idx}
            className={`p-5 rounded-2xl bg-white dark:bg-slate-900 border ${card.borderColor} shadow-sm hover:shadow-md transition-all flex flex-col justify-between`}
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">
                {card.title}
              </span>
              <div className={`p-2 rounded-xl bg-slate-50 dark:bg-slate-800 ${card.textColor}`}>
                <IconComponent className="w-4 h-4" />
              </div>
            </div>

            <div className="mt-3">
              <div className="text-2xl font-extrabold text-slate-900 dark:text-white tracking-tight">
                {card.value}
              </div>
              <div className="mt-1 flex items-center gap-1 text-[11px] text-slate-500 dark:text-slate-400 font-medium">
                <ArrowUpRight className="w-3 h-3 text-emerald-500" />
                <span>{card.change}</span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

