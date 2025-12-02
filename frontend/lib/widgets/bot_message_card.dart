// lib/widgets/bot_message_card.dart

import 'package:flutter/material.dart';
import '../models/predict_result.dart';

class BotMessageCard extends StatelessWidget {
  final PredictResult result;

  const BotMessageCard({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: result.labels
                  .map(
                    (lab) => Chip(
                      label: Text(lab),
                      backgroundColor: theme.colorScheme.secondaryContainer,
                      labelStyle: theme.textTheme.bodyMedium,
                      materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                    ),
                  )
                  .toList(),
            ),
            const SizedBox(height: 8),
            Text(
              "Probs:",
              style: theme.textTheme.bodyMedium!
                  .copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 4),
            if (result.probs.isNotEmpty)
              Text(
                result.probs.entries
                    .map(
                      (e) => "${e.key}: ${e.value.toStringAsFixed(2)}",
                    )
                    .join("   "),
                style: theme.textTheme.bodySmall,
              )
            else
              Text(
                "(not available)",
                style: theme.textTheme.bodySmall,
              ),
            const SizedBox(height: 8),
            Text(
              "Advice:",
              style: theme.textTheme.bodyMedium!
                  .copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 4),
            Text(
              result.advice.isNotEmpty
                  ? result.advice
                  : "No specific advice returned.",
              style: theme.textTheme.bodyMedium,
            ),
          ],
        ),
      ),
    );
  }
}
