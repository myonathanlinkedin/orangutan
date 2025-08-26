#!/usr/bin/env python3
"""
ORANGUTAN Negotiation Engine
Implements the negotiation priority function from orangutan.txt specification
"""

import time
from typing import Dict, List, Tuple
import numpy as np

from ..env.jungle import Jungle


class NegotiationEngine:
    """
    ORANGUTAN Negotiation Engine implementing the priority function:
    Π_a = α · (p_a / max_p) + β · (1/L_a) / max(1/L) + γ · U_recent(a)
    """
    
    def __init__(self, jungle: Jungle):
        self.jungle = jungle
        self.negotiation_history = []
        
        # Priority function coefficients (α, β, γ)
        self.alpha = 0.4  # Priority weight
        self.beta = 0.4   # SLO urgency weight  
        self.gamma = 0.2  # Recent utility weight
        
        # Performance tracking
        self.agent_performance = {}
        self.negotiation_rounds = 0
    
    def negotiate_assignments(
        self, 
        proposals: List[Dict], 
        current_occupancy: Dict[int, float]
    ) -> Dict[int, List[Dict]]:
        """
        Negotiate workload assignments to SMs using priority function.
        
        Args:
            proposals: List of workload proposals with (SM, tile) candidates
            current_occupancy: Current SM occupancy levels
            
        Returns:
            Dictionary mapping SM IDs to assigned workloads
        """
        self.negotiation_rounds += 1
        
        # Calculate priority scores for all proposals
        scored_proposals = self._calculate_priority_scores(proposals)
        
        # Sort by priority score (highest first)
        scored_proposals.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Assign workloads to SMs using greedy approach
        assignments = self._assign_workloads_to_sms(scored_proposals, current_occupancy)
        
        # Record negotiation history
        self._record_negotiation_round(proposals, scored_proposals, assignments)
        
        return assignments
    
    def _calculate_priority_scores(self, proposals: List[Dict]) -> List[Dict]:
        """Calculate priority scores using the ORANGUTAN priority function."""
        if not proposals:
            return []
        
        # Extract all priorities and SLOs for normalization
        priorities = [p.get('priority', 1) for p in proposals]
        slos = [p.get('slo_ms', 100) for p in proposals]
        
        max_priority = max(priorities) if priorities else 1
        max_slo_factor = max(1.0 / slo if slo > 0 else 0 for slo in slos) if slos else 1.0
        
        scored_proposals = []
        
        for proposal in proposals:
            workload_id = proposal['workload_id']
            priority = proposal.get('priority', 1)
            slo_ms = proposal.get('slo_ms', 100)
            
            # Calculate priority components
            priority_component = (priority / max_priority) if max_priority > 0 else 0
            slo_component = (1.0 / slo_ms) / max_slo_factor if slo_ms > 0 and max_slo_factor > 0 else 0
            utility_component = self._get_recent_utility(workload_id)
            
            # Apply priority function: Π_a = α · (p_a / max_p) + β · (1/L_a) / max(1/L) + γ · U_recent(a)
            priority_score = (
                self.alpha * priority_component +
                self.beta * slo_component +
                self.gamma * utility_component
            )
            
            scored_proposal = {
                **proposal,
                'priority_score': priority_score,
                'priority_component': priority_component,
                'slo_component': slo_component,
                'utility_component': utility_component
            }
            
            scored_proposals.append(scored_proposal)
        
        return scored_proposals
    
    def _get_recent_utility(self, workload_id: str) -> float:
        """Get recent utility (throughput gain) for a workload."""
        if workload_id in self.agent_performance:
            recent_performance = self.agent_performance[workload_id]
            if recent_performance:
                # Calculate utility as recent throughput improvement
                return np.mean(recent_performance[-5:])  # Last 5 measurements
        return 0.5  # Default neutral utility
    
    def _assign_workloads_to_sms(
        self, 
        scored_proposals: List[Dict], 
        current_occupancy: Dict[int, float]
    ) -> Dict[int, List[Dict]]:
        """
        Assign workloads to SMs using greedy approach with fallback levels.
        Implements the fallback mechanism from orangutan.txt specification.
        """
        assignments = {}
        unassigned_proposals = scored_proposals.copy()
        
        # First pass: assign to preferred SMs
        for proposal in scored_proposals:
            preferred_sm = proposal.get('preferred_sm')
            if preferred_sm and self._can_assign_to_sm(proposal, preferred_sm, current_occupancy):
                self._assign_proposal_to_sm(proposal, preferred_sm, assignments, current_occupancy)
                unassigned_proposals.remove(proposal)
        
        # Second pass: assign remaining proposals with fallback
        for proposal in unassigned_proposals:
            assigned = False
            
            # Try primary SM
            primary_sm = proposal.get('primary_sm')
            if primary_sm and self._can_assign_to_sm(proposal, primary_sm, current_occupancy):
                self._assign_proposal_to_sm(proposal, primary_sm, assignments, current_occupancy)
                assigned = True
                continue
            
            # Try fallback SMs
            fallback_sms = proposal.get('fallback_sms', [])
            for fallback_sm in fallback_sms:
                if self._can_assign_to_sm(proposal, fallback_sm, current_occupancy):
                    self._assign_proposal_to_sm(proposal, fallback_sm, assignments, current_occupancy)
                    assigned = True
                    break
            
            # If still unassigned, try any available SM
            if not assigned:
                available_sm = self._find_available_sm(proposal, current_occupancy)
                if available_sm:
                    self._assign_proposal_to_sm(proposal, available_sm, assignments, current_occupancy)
                else:
                    # Workload cannot be assigned - will be retried in next round
                    pass
        
        return assignments
    
    def _can_assign_to_sm(self, proposal: Dict, sm_id: int, current_occupancy: Dict[int, float]) -> bool:
        """Check if a proposal can be assigned to a specific SM."""
        # Check SM occupancy
        sm_occupancy = current_occupancy.get(sm_id, 0.0)
        if sm_occupancy >= 1.0:  # SM is fully occupied
            return False
        
        # Check resource constraints
        tile_shape = proposal.get('tile_shape', (64, 64, 32))
        m, n, k = tile_shape
        
        # Check register budget (simplified)
        register_usage = m * n * k / (64 * 1024 * 1024)  # Normalized to SM capacity
        if register_usage > 0.8:  # Leave 20% margin
            return False
        
        # Check shared memory budget
        shared_mem_usage = m * n / (192 * 1024)  # Normalized to SM capacity
        if shared_mem_usage > 0.8:  # Leave 20% margin
            return False
        
        return True
    
    def _assign_proposal_to_sm(
        self, 
        proposal: Dict, 
        sm_id: int, 
        assignments: Dict[int, List[Dict]], 
        current_occupancy: Dict[int, float]
    ):
        """Assign a proposal to an SM and update occupancy."""
        if sm_id not in assignments:
            assignments[sm_id] = []
        
        assignments[sm_id].append(proposal)
        
        # Update occupancy
        tile_shape = proposal.get('tile_shape', (64, 64, 32))
        m, n, k = tile_shape
        
        # Calculate occupancy increase
        register_usage = m * n * k / (64 * 1024 * 1024)
        shared_mem_usage = m * n / (192 * 1024)
        
        occupancy_increase = max(register_usage, shared_mem_usage)
        current_occupancy[sm_id] = current_occupancy.get(sm_id, 0.0) + occupancy_increase
    
    def _find_available_sm(self, proposal: Dict, current_occupancy: Dict[int, float]) -> int:
        """Find an available SM for a proposal."""
        # Find SM with lowest occupancy
        available_sms = [
            sm_id for sm_id, occupancy in current_occupancy.items()
            if occupancy < 0.8  # Leave 20% margin
        ]
        
        if available_sms:
            return min(available_sms, key=lambda sm: current_occupancy.get(sm, 0.0))
        
        return None
    
    def _record_negotiation_round(
        self, 
        original_proposals: List[Dict], 
        scored_proposals: List[Dict], 
        assignments: Dict[int, List[Dict]]
    ):
        """Record negotiation round for analysis and learning."""
        round_record = {
            'round': self.negotiation_rounds,
            'timestamp': time.time(),
            'total_proposals': len(original_proposals),
            'assigned_proposals': sum(len(workloads) for workloads in assignments.values()),
            'priority_distribution': {
                'min': min(p['priority_score'] for p in scored_proposals) if scored_proposals else 0,
                'max': max(p['priority_score'] for p in scored_proposals) if scored_proposals else 0,
                'mean': np.mean([p['priority_score'] for p in scored_proposals]) if scored_proposals else 0
            },
            'sm_assignments': {
                sm_id: len(workloads) for sm_id, workloads in assignments.items()
            }
        }
        
        self.negotiation_history.append(round_record)
    
    def update_agent_performance(self, workload_id: str, performance_metric: float):
        """Update performance tracking for an agent/workload."""
        if workload_id not in self.agent_performance:
            self.agent_performance[workload_id] = []
        
        self.agent_performance[workload_id].append(performance_metric)
        
        # Keep only recent performance (last 20 measurements)
        if len(self.agent_performance[workload_id]) > 20:
            self.agent_performance[workload_id] = self.agent_performance[workload_id][-20:]
    
    def get_negotiation_statistics(self) -> Dict:
        """Get negotiation statistics for analysis."""
        if not self.negotiation_history:
            return {}
        
        total_rounds = len(self.negotiation_history)
        total_proposals = sum(round_data['total_proposals'] for round_data in self.negotiation_history)
        total_assignments = sum(round_data['assigned_proposals'] for round_data in self.negotiation_history)
        
        return {
            'total_negotiation_rounds': total_rounds,
            'total_proposals_processed': total_proposals,
            'total_assignments_made': total_assignments,
            'assignment_rate': total_assignments / total_proposals if total_proposals > 0 else 0,
            'average_priority_score': np.mean([
                round_data['priority_distribution']['mean'] 
                for round_data in self.negotiation_history
            ]),
            'negotiation_efficiency': total_assignments / total_rounds if total_rounds > 0 else 0
        }
    
    def optimize_priority_coefficients(self):
        """Optimize priority function coefficients based on performance."""
        if len(self.negotiation_history) < 10:
            return  # Need more data for optimization
        
        # Simple optimization: adjust coefficients based on assignment success rate
        recent_rounds = self.negotiation_history[-10:]
        success_rates = [
            round_data['assigned_proposals'] / round_data['total_proposals']
            for round_data in recent_rounds
            if round_data['total_proposals'] > 0
        ]
        
        if success_rates:
            avg_success_rate = np.mean(success_rates)
            
            # If success rate is low, increase flexibility (reduce priority weight)
            if avg_success_rate < 0.7:
                self.alpha = max(0.2, self.alpha * 0.9)
                self.beta = min(0.6, self.beta * 1.1)
                self.gamma = min(0.3, self.gamma * 1.1)
            
            # If success rate is high, increase optimization (increase priority weight)
            elif avg_success_rate > 0.9:
                self.alpha = min(0.6, self.alpha * 1.1)
                self.beta = max(0.2, self.beta * 0.9)
                self.gamma = max(0.1, self.gamma * 0.9)
    
    def reset_negotiation_state(self):
        """Reset negotiation state for new simulation."""
        self.negotiation_history = []
        self.negotiation_rounds = 0
        self.agent_performance = {}
