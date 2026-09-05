// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2023 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include "UnitTestFramework.h"
#include "PhysicsTestContext.h"
#include <Jolt/Core/JobSystemSingleThreaded.h>
#include <Jolt/Physics/Constraints/DistanceConstraint.h>
#include <Jolt/Physics/PhysicsStepListener.h>
#include "Layers.h"

namespace
{
	// Force the legal scheduling order in which ready gravity jobs run last.
	// This exposes missing setup dependencies without relying on a timing race.
	class GravityLastJobSystem final : public JobSystem
	{
	public:
		explicit GravityLastJobSystem(bool inDelayGravity) : mDelayGravity(inDelayGravity), mInner(cMaxPhysicsJobs) { }
		int GetMaxConcurrency() const override { return 1; }
		JobHandle CreateJob(const char *inName, ColorArg inColor, const JobFunction &inFunction, uint32 inDependencies = 0) override
		{
			const bool delay = mDelayGravity && strcmp(inName, "ApplyGravity") == 0;
			JobHandle job = mInner.CreateJob(inName, inColor, inFunction, inDependencies + (delay? 1 : 0));
			mJobs.push_back(job);
			if (delay)
				mDelayedGravity.push_back(job);
			return job;
		}
		Barrier *CreateBarrier() override { return mInner.CreateBarrier(); }
		void DestroyBarrier(Barrier *inBarrier) override { mInner.DestroyBarrier(inBarrier); }
		void WaitForJobs(Barrier *inBarrier) override
		{
			for (const JobHandle &job : mDelayedGravity)
				job.RemoveDependency();
			mInner.WaitForJobs(inBarrier);
			for (const JobHandle &job : mJobs)
				CHECK(job.IsDone());
			mDelayedGravity.clear();
			mJobs.clear();
		}
	protected:
		// Jobs belong to mInner, so these callbacks must never reach this wrapper.
		void QueueJob(Job *) override { FAIL("Unexpected wrapper QueueJob"); }
		void QueueJobs(Job **, uint) override { FAIL("Unexpected wrapper QueueJobs"); }
		void FreeJob(Job *) override { FAIL("Unexpected wrapper FreeJob"); }
	private:
		bool mDelayGravity;
		JobSystemSingleThreaded mInner;
		Array<JobHandle> mJobs;
		Array<JobHandle> mDelayedGravity;
	};
}

TEST_SUITE("DistanceConstraintTests")
{
	TEST_CASE("TestDistanceVelocityBiasUsesPostForceSnapshot")
	{
		for (int substeps : { 1, 2 })
			for (bool with_listener : { false, true })
			{
				CAPTURE(substeps);
				CAPTURE(with_listener);
				RVec3 reference_position = RVec3::sZero();
				Vec3 reference_velocity = Vec3::sZero();
				// Immediate execution, gravity deliberately last, and real workers.
				for (int schedule : { 0, 1, 2 })
				{
					CAPTURE(schedule);
					PhysicsTestContext context(1.0f / 22.0f, substeps, schedule == 2? 4 : 0);
					context.ZeroGravity();
					PhysicsSettings physics_settings = context.GetSystem()->GetPhysicsSettings();
					physics_settings.mNumVelocitySteps = 10;
					physics_settings.mNumPositionSteps = 0;
					context.GetSystem()->SetPhysicsSettings(physics_settings);
					Body &body = context.CreateSphere(RVec3(9, 0, 0), 0.5f, EMotionType::Dynamic, EMotionQuality::Discrete, Layers::MOVING);
					body.GetMotionProperties()->SetLinearDamping(0.0f);
					const float step_dt = context.GetStepDeltaTime();
					body.AddForce(Vec3(2.0f / (step_dt * step_dt * body.GetMotionProperties()->GetInverseMass()), 0, 0));

					DistanceConstraintSettings settings;
					settings.mPoint2 = body.GetPosition();
					settings.mMinDistance = 0.0f;
					settings.mMaxDistance = 10.0f;
					DistanceConstraint &constraint = context.CreateConstraint<DistanceConstraint>(Body::sFixedToWorld, body, settings);
					constraint.SetLimitsVelocityBias(1.0f, 0.5f);
					if (schedule == 2)
						for (int i = 0; i < 256; ++i)
							context.CreateSphere(RVec3(2.0f * (i % 16), 10.0f + 2.0f * (i / 16), 0), 0.1f, EMotionType::Dynamic, EMotionQuality::Discrete, Layers::MOVING2);

					struct StepListener : PhysicsStepListener
					{
						void OnStep(const PhysicsStepListenerContext &) override { ++mCalls; }
						int mCalls = 0;
					} listener;
					if (with_listener)
						context.GetSystem()->AddStepListener(&listener);
					GravityLastJobSystem jobs(schedule == 1);
					CHECK(context.GetSystem()->Update(context.GetDeltaTime(), substeps, context.GetTempAllocator(), schedule == 2? context.GetJobSystem() : &jobs) == EPhysicsUpdateError::None);
					if (with_listener)
					{
						context.GetSystem()->RemoveStepListener(&listener);
						CHECK(listener.mCalls == substeps);
					}
					if (schedule == 0)
					{
						reference_position = body.GetPosition();
						reference_velocity = body.GetLinearVelocity();
					}
					else
					{
						CHECK_APPROX_EQUAL(reference_position, body.GetPosition(), 1.0e-5_r);
						CHECK_APPROX_EQUAL(reference_velocity, body.GetLinearVelocity(), 1.0e-4f);
					}
				}
			}
	}

	// Test if the distance constraint can be used to create a spring
	TEST_CASE("TestDistanceSpring")
	{
		// Configuration of the spring
		const RVec3 cInitialPosition(10, 0, 0);
		const float cFrequency = 2.0f;
		const float cDamping = 0.1f;

		for (int mode = 0; mode < 2; ++mode)
		{
			// Create a sphere
			PhysicsTestContext context;
			context.ZeroGravity();
			Body &body = context.CreateSphere(cInitialPosition, 0.5f, EMotionType::Dynamic, EMotionQuality::Discrete, Layers::MOVING);
			body.GetMotionProperties()->SetLinearDamping(0.0f);

			// Calculate stiffness and damping of spring
			float m = 1.0f / body.GetMotionProperties()->GetInverseMass();
			float omega = 2.0f * JPH_PI * cFrequency;
			float k = m * Square(omega);
			float c = 2.0f * m * cDamping * omega;

			// Create spring
			DistanceConstraintSettings constraint;
			constraint.mPoint2 = cInitialPosition;
			if (mode == 0)
			{
				// First iteration use stiffness and damping
				constraint.mLimitsSpringSettings.mMode = ESpringMode::StiffnessAndDamping;
				constraint.mLimitsSpringSettings.mStiffness = k;
				constraint.mLimitsSpringSettings.mDamping = c;
			}
			else
			{
				// Second iteration use frequency and damping
				constraint.mLimitsSpringSettings.mMode = ESpringMode::FrequencyAndDamping;
				constraint.mLimitsSpringSettings.mFrequency = cFrequency;
				constraint.mLimitsSpringSettings.mDamping = cDamping;
			}
			constraint.mMinDistance = constraint.mMaxDistance = 0.0f;
			context.CreateConstraint<DistanceConstraint>(Body::sFixedToWorld, body, constraint);

			// Simulate spring
			Real x = cInitialPosition.GetX();
			float v = 0.0f;
			float dt = context.GetDeltaTime();
			for (int i = 0; i < 120; ++i)
			{
				// Using the equations from page 32 of Soft Constraints: Reinventing The Spring - Erin Catto - GDC 2011 for an implicit euler spring damper
				v = (v - dt * k / m * float(x)) / (1.0f + dt * c / m + Square(dt) * k / m);
				x += v * dt;

				// Run physics simulation
				context.SimulateSingleStep();

				// Test if simulation matches prediction
				CHECK_APPROX_EQUAL(x, body.GetPosition().GetX(), 5.0e-6_r);
				CHECK(body.GetPosition().GetY() == 0);
				CHECK(body.GetPosition().GetZ() == 0);
			}
		}
	}

	TEST_CASE("TestDistanceVelocityBiasIsIndependentOfSolverIterations")
	{
		const uint cVelocitySteps[] = { 1, 4, 10 };
		for (uint num_velocity_steps : cVelocitySteps)
		{
			PhysicsTestContext context;
			context.ZeroGravity();
			PhysicsSettings physics_settings = context.GetSystem()->GetPhysicsSettings();
			physics_settings.mNumVelocitySteps = num_velocity_steps;
			physics_settings.mNumPositionSteps = 0;
			context.GetSystem()->SetPhysicsSettings(physics_settings);

			Body &body = context.CreateSphere(RVec3(12, 0, 0), 0.5f, EMotionType::Dynamic, EMotionQuality::Discrete, Layers::MOVING);
			body.GetMotionProperties()->SetLinearDamping(0.0f);

			DistanceConstraintSettings constraint;
			constraint.mPoint1 = RVec3::sZero();
			constraint.mPoint2 = body.GetPosition();
			constraint.mMinDistance = 0.0f;
			constraint.mMaxDistance = 10.0f;
			DistanceConstraint &distance_constraint = context.CreateConstraint<DistanceConstraint>(Body::sFixedToWorld, body, constraint);
			distance_constraint.SetLimitsVelocityBias(1.0f, 0.5f);

			context.SimulateSingleStep();

			CHECK_APPROX_EQUAL(10.0_r, body.GetPosition().GetX(), 1.0e-5_r);
			CHECK_APPROX_EQUAL(-120.0f, body.GetLinearVelocity().GetX(), 1.0e-4f);
		}
	}

	TEST_CASE("TestDistanceVelocityBiasPredictsLimitCrossing")
	{
		PhysicsTestContext context;
		context.ZeroGravity();
		PhysicsSettings physics_settings = context.GetSystem()->GetPhysicsSettings();
		physics_settings.mNumVelocitySteps = 4;
		physics_settings.mNumPositionSteps = 0;
		context.GetSystem()->SetPhysicsSettings(physics_settings);

		Body &body = context.CreateSphere(RVec3(9, 0, 0), 0.5f, EMotionType::Dynamic, EMotionQuality::Discrete, Layers::MOVING);
		body.GetMotionProperties()->SetLinearDamping(0.0f);
		body.SetLinearVelocity(Vec3(120, 0, 0));

		DistanceConstraintSettings constraint;
		constraint.mPoint1 = RVec3::sZero();
		constraint.mPoint2 = body.GetPosition();
		constraint.mMinDistance = 0.0f;
		constraint.mMaxDistance = 10.0f;
		DistanceConstraint &distance_constraint = context.CreateConstraint<DistanceConstraint>(Body::sFixedToWorld, body, constraint);
		distance_constraint.SetLimitsVelocityBias(1.0f, 0.5f);

		context.SimulateSingleStep();

		CHECK_APPROX_EQUAL(10.0_r, body.GetPosition().GetX(), 1.0e-5_r);
		CHECK_APPROX_EQUAL(60.0f, body.GetLinearVelocity().GetX(), 1.0e-4f);
	}

	TEST_CASE("TestDistanceVelocityBiasDampsRigidConstraintAtLiveSolverRate")
	{
		PhysicsTestContext context(1.0f / 22.0f, 2);
		context.ZeroGravity();
		PhysicsSettings physics_settings = context.GetSystem()->GetPhysicsSettings();
		physics_settings.mNumVelocitySteps = 10;
		physics_settings.mNumPositionSteps = 0;
		context.GetSystem()->SetPhysicsSettings(physics_settings);

		Body &body = context.CreateSphere(RVec3(12, 0, 0), 0.5f, EMotionType::Dynamic, EMotionQuality::Discrete, Layers::MOVING);
		body.GetMotionProperties()->SetLinearDamping(0.0f);

		DistanceConstraintSettings constraint;
		constraint.mPoint1 = RVec3::sZero();
		constraint.mPoint2 = body.GetPosition();
		constraint.mMinDistance = 10.0f;
		constraint.mMaxDistance = 10.0f;
		DistanceConstraint &distance_constraint = context.CreateConstraint<DistanceConstraint>(Body::sFixedToWorld, body, constraint);
		distance_constraint.SetLimitsVelocityBias(1.0f, 0.5f);

		for (int step = 0; step < 12; ++step)
			context.SimulateSingleStep();

		CHECK_APPROX_EQUAL(10.0_r, body.GetPosition().GetX(), 1.0e-2_r);
		CHECK_APPROX_EQUAL(0.0f, body.GetLinearVelocity().GetX(), 1.0e-1f);
	}
}
